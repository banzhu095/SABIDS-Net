from __future__ import annotations

import math
import hashlib
import json
import time
import warnings
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import cv2
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from torch.utils.checkpoint import checkpoint
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # Training metrics still persist in CSV/JSON without it.
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            warnings.warn(
                "tensorboard is unavailable; continuing with CSV/JSON logging only.",
                RuntimeWarning,
            )

        def add_scalar(self, *args, **kwargs) -> None:
            return None

        def close(self) -> None:
            return None
from tqdm import tqdm

from ..data import GroupUniformSampler, OCTManifestDataset, SparseAnnotationSampler
from ..data.transforms import JointOCTTransform
from ..losses import SABIDSLoss
from ..training import PhaseStateMachine
from ..metrics import binary_metrics, soft_dice_score, vessel_diagnostic_metrics
from ..models import ModelEMA, SABIDSNet
from ..utils import (
    CSVLogger,
    count_parameters,
    get_device,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    write_json,
)
from ..experiments.protocol_lock import (
    load_protocol_lock,
    validate_checkpoint_config,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_label_asset(path: Path, allow_float_cache: bool = False) -> Dict[str, object]:
    raw_sha256 = _sha256_file(path)
    if allow_float_cache and path.suffix.lower() == ".npy":
        decoded = np.load(path, allow_pickle=False)
    else:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise RuntimeError(f"OpenCV failed to decode label asset: {path}")
    decoded = np.ascontiguousarray(decoded)
    content_digest = hashlib.sha256()
    content_digest.update(str(decoded.dtype).encode("utf-8"))
    content_digest.update(str(tuple(decoded.shape)).encode("utf-8"))
    content_digest.update(decoded.tobytes())
    values, counts = np.unique(decoded, return_counts=True)
    return {
        "raw_sha256": raw_sha256,
        "decoded_sha256": content_digest.hexdigest(),
        "shape": list(decoded.shape),
        "dtype": str(decoded.dtype),
        "value_counts": {
            str(value.item() if hasattr(value, "item") else value): int(count)
            for value, count in zip(values, counts)
        },
    }


def _make_transform(config: Dict, training: bool) -> JointOCTTransform:
    size = config["data"].get("target_size", [512, 1024])
    augmentation = config["data"].get("augmentation", {})
    return JointOCTTransform(
        target_size=(int(size[0]), int(size[1])),
        training=training,
        horizontal_flip=float(augmentation.get("horizontal_flip", 0.5 if training else 0.0)),
        normalization=config["data"].get("normalization", "fixed"),
        percentile_low=float(config["data"].get("percentile_low", 0.5)),
        percentile_high=float(config["data"].get("percentile_high", 99.5)),
        strong_private_only=bool(augmentation.get("strong_private_only", True)),
        gamma_range=tuple(augmentation.get("gamma_range", [0.8, 1.2])),
        contrast_range=tuple(augmentation.get("contrast_range", [0.85, 1.15])),
        speckle_std=float(augmentation.get("speckle_std", 0.03)),
        blur_probability=float(augmentation.get("blur_probability", 0.1)),
    )


def _load_segmentation_labels(config: Dict) -> bool:
    """Resolve label I/O without changing non-denoising training behavior."""
    data_cfg = config["data"]
    configured = data_cfg.get("load_segmentation_labels")
    if configured is not None:
        return bool(configured)
    return str(config.get("train", {}).get("stage", "joint")).lower() != "denoise"


def build_loaders(config: Dict) -> tuple[DataLoader, DataLoader, object]:
    data_cfg = config["data"]
    load_segmentation_labels = _load_segmentation_labels(config)
    train_dataset = OCTManifestDataset(
        data_cfg["manifest"],
        split=data_cfg.get("train_split", "train"),
        transform=_make_transform(config, True),
        sample_repeat=not config.get("dose_response", {}).get("enabled", False),
        root=data_cfg.get("root"),
        datasets=data_cfg.get("train_datasets"),
        groups=data_cfg.get("train_groups"),
        image_column=data_cfg.get("input_column", "image_path"),
        guidance_mapping=data_cfg.get("guidance_mapping"),
        load_segmentation_labels=load_segmentation_labels,
        pretransformed_model_grid=bool(data_cfg.get("pretransformed_model_grid", False)),
        deterministic_augmentation_seed=(int(config.get("seed", 42))
            if data_cfg.get("deterministic_augmentation", False) else None),
    )
    val_dataset = OCTManifestDataset(
        data_cfg["manifest"],
        split=data_cfg.get("val_split", "val"),
        transform=_make_transform(config, False),
        sample_repeat=False,
        root=data_cfg.get("root"),
        datasets=data_cfg.get("val_datasets"),
        groups=data_cfg.get("val_groups"),
        image_column=data_cfg.get("input_column", "image_path"),
        guidance_mapping=data_cfg.get("guidance_mapping"),
        load_segmentation_labels=load_segmentation_labels,
        pretransformed_model_grid=bool(data_cfg.get("pretransformed_model_grid", False)),
    )
    max_val_samples = data_cfg.get("max_val_samples")
    if max_val_samples is not None:
        count = min(int(max_val_samples), len(val_dataset))
        val_dataset = Subset(val_dataset, range(count))
    samples_per_epoch = data_cfg.get("samples_per_epoch")
    if data_cfg.get("max_train_samples") is not None:
        samples_per_epoch = min(
            int(data_cfg["max_train_samples"]),
            int(samples_per_epoch or len(train_dataset)),
        )
    vessel_fraction = float(data_cfg.get("vessel_oversample_fraction", 0.0) or 0.0)
    if vessel_fraction > 0.0:
        train_sampler = SparseAnnotationSampler(
            train_dataset,
            vessel_fraction=vessel_fraction,
            samples_per_epoch=samples_per_epoch,
            seed=int(config.get("seed", 42)),
        )
    else:
        train_sampler = GroupUniformSampler(
            train_dataset,
            samples_per_epoch=samples_per_epoch,
            seed=int(config.get("seed", 42)),
        )
    data_seed = int(config.get("seed", 42)) + 1_000_003
    train_generator = torch.Generator().manual_seed(data_seed)
    val_generator = torch.Generator().manual_seed(data_seed + 1)
    loader_args = {
        "batch_size": int(config["train"].get("batch_size", 2)),
        "num_workers": int(config["train"].get("num_workers", 4)),
        "pin_memory": True,
        "persistent_workers": int(config["train"].get("num_workers", 4)) > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        drop_last=True,
        generator=train_generator,
        **loader_args,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        generator=val_generator,
        **loader_args,
    )
    return train_loader, val_loader, train_sampler


def build_diagnostic_loader(
    config: Dict, split: str, max_samples: Optional[int] = None
) -> DataLoader:
    data_cfg = config["data"]
    dataset = OCTManifestDataset(
        data_cfg["manifest"],
        split=split,
        transform=_make_transform(config, False),
        sample_repeat=False,
        root=data_cfg.get("root"),
        datasets=(
            data_cfg.get("train_datasets")
            if split == data_cfg.get("train_split", "train")
            else data_cfg.get("val_datasets")
        ),
        groups=(
            data_cfg.get("train_groups")
            if split == data_cfg.get("train_split", "train")
            else data_cfg.get("val_groups")
        ),
        image_column=data_cfg.get("input_column", "image_path"),
        guidance_mapping=data_cfg.get("guidance_mapping"),
        load_segmentation_labels=_load_segmentation_labels(config),
    )
    frames_per_group = config["data"].get("train_eval_frames_per_group")
    if split == data_cfg.get("train_split", "train") and frames_per_group is not None:
        indices = []
        for group_id in sorted(dataset.groups):
            indices.extend(dataset.groups[group_id][: int(frames_per_group)])
        dataset = Subset(dataset, indices)
    if max_samples is not None:
        dataset = Subset(dataset, range(min(int(max_samples), len(dataset))))
    workers = int(config.get("evaluation", {}).get("num_workers", 2))
    return DataLoader(
        dataset,
        batch_size=int(config.get("evaluation", {}).get("batch_size", 1)),
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def build_model(config: Dict) -> SABIDSNet:
    model_cfg = config["model"]
    return SABIDSNet(
        in_channels=int(model_cfg.get("in_channels", 1)),
        channels=tuple(model_cfg.get("channels", [32, 64, 128, 256])),
        encoder_depths=tuple(model_cfg.get("encoder_depths", [2, 2, 4, 6])),
        decoder_depth=int(model_cfg.get("decoder_depth", 2)),
        interaction_levels=tuple(model_cfg.get("interaction_levels", [3, 2, 1])),
        enable_seg_to_denoise=bool(model_cfg.get("s2d_enabled", model_cfg.get("enable_seg_to_denoise", True))),
        enable_denoise_to_seg=bool(model_cfg.get("d2s_enabled", model_cfg.get("enable_denoise_to_seg", True))),
        use_uncertainty=bool(model_cfg.get("use_uncertainty", True)),
        detach_denoise_to_seg_source=bool(
            model_cfg.get("detach_d2s_source", model_cfg.get("detach_denoise_to_seg_source", False))
        ),
        dropout=float(model_cfg.get("dropout", 0.0)),
        residual_scale=float(model_cfg.get("residual_scale", 0.5)),
        causal_interaction_experiment=bool(model_cfg.get("causal_interaction_experiment", False)),
        detach_seg_to_denoise_source=bool(model_cfg.get("detach_s2d_source", False)),
        interaction_scale_init=float(model_cfg.get("interaction_scale_init", 0.1)),
        s2d_source_mode=str(model_cfg.get("s2d_source_mode", "cross")),
        d2s_source_mode=str(model_cfg.get("d2s_source_mode", "cross")),
        strong_s2d_rho=model_cfg.get("strong_s2d_rho"),
        strong_d2s_rho=model_cfg.get("strong_d2s_rho"),
    )


def _effective_dataset_table(dataset: object) -> pd.DataFrame:
    """Return exactly the rows exposed by a dataset, including Subset filters."""
    if isinstance(dataset, Subset):
        base = _effective_dataset_table(dataset.dataset)
        indices = [int(index) for index in dataset.indices]
        return base.iloc[indices].reset_index(drop=True)
    table = getattr(dataset, "table", None)
    if not isinstance(table, pd.DataFrame):
        raise TypeError("Training asset evidence requires a manifest-backed dataset")
    return table.copy().reset_index(drop=True)


class Trainer:
    def __init__(self, config: Dict):
        if config.get("dose_response", {}).get("enabled", False):
            from sabids.experiments.dose_response import validate_dose_config
            validate_dose_config(config)
        self.config = config
        self.device = get_device(config.get("device", "auto"))
        seed_everything(
            int(config.get("seed", 42)),
            bool(config.get("deterministic", False)),
            use_cuda=self.device.type == "cuda",
        )
        self.output_dir = Path(config["train"].get("output_dir", "runs/sabids"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._training_asset_initial_path: Optional[Path] = None
        self._training_asset_filtered: Optional[pd.DataFrame] = None
        self._prepare_formal_d2_teacher_config()
        self._prepare_training_asset_evidence_config()
        self.train_loader, self.val_loader, self.train_sampler = build_loaders(config)
        self._record_run_inputs()
        self._record_training_asset_evidence()
        if str(config.get("train", {}).get("stage", "")) in {"interaction", "input_segment"}:
            missing_labels = config.get("runtime", {}).get("missing_label_assets", [])
            if missing_labels:
                raise FileNotFoundError(
                    "Referenced train/validation label assets are missing: "
                    + ", ".join(str(path) for path in missing_labels[:10])
                )
        train_eval_every = int(config["train"].get("train_eval_every", 0) or 0)
        self.train_eval_every = train_eval_every
        self.train_eval_loader = (
            build_diagnostic_loader(
                config,
                split=config["data"].get("train_split", "train"),
                max_samples=config["data"].get("max_train_eval_samples"),
            )
            if train_eval_every > 0
            else None
        )
        self.model = build_model(config).to(self.device)
        self.stage = config["train"].get("stage", "joint")
        self.model.set_train_stage(
            self.stage,
            private_train_encoder_levels=config.get("model", {}).get(
                "private_train_encoder_levels", []
            ),
            freeze_shared_encoder=bool(
                config.get("model", {}).get(
                    "freeze_shared_encoder",
                    config.get("model", {}).get("stage2_freeze_shared_encoder", False),
                )
            ),
            train_denoise_to_seg=bool(
                config.get("model", {}).get("stage2_train_denoise_to_seg", False)
            ),
        )
        self.d2_teacher: Optional[SABIDSNet] = None
        self._d2_teacher_initial_sha: Dict[str, str] = {}
        self._setup_d2_teacher()
        self.loss_fn = SABIDSLoss(config["loss"]).to(self.device)
        memory_safe_joint = bool(
            config["train"].get("memory_safe_joint", True)
        )
        clean_teacher_no_grad = bool(
            config["train"].get("clean_teacher_no_grad", True)
        )
        identity_weight = float(
            config.get("loss", {}).get("weights", {}).get("identity", 0.0)
        )
        if (
            self.stage in {"joint", "private"}
            and memory_safe_joint
            and clean_teacher_no_grad
            and identity_weight > 0.0
        ):
            raise ValueError(
                "Memory-safe joint training uses the clean image as a stop-gradient "
                "RMAC teacher, so loss.weights.identity must be 0. Set identity=0 "
                "or disable train.clean_teacher_no_grad."
            )
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=float(config["train"].get("learning_rate", 2e-4)),
            weight_decay=float(config["train"].get("weight_decay", 1e-4)),
        )
        write_json(
            {
                "trainable": [
                    name
                    for name, parameter in self.model.named_parameters()
                    if parameter.requires_grad
                ],
                "frozen": [
                    name
                    for name, parameter in self.model.named_parameters()
                    if not parameter.requires_grad
                ],
                "optimizer_parameter_count": int(
                    sum(
                        parameter.numel()
                        for group in self.optimizer.param_groups
                        for parameter in group["params"]
                    )
                ),
            },
            self.output_dir / "parameter_audit.json",
        )
        epochs = int(config["train"].get("epochs", 100))
        schedule = config["train"].get("schedule")
        self.scheduler_step_per_optimizer = schedule in {
            "order_ds", "order_sd", "order_alt"
        }
        scheduler_name = str(config["train"].get("scheduler", "cosine"))
        self.scheduler_name = scheduler_name
        if self.scheduler_step_per_optimizer and scheduler_name == "plateau":
            raise ValueError("Continuous order schedules require the global-step cosine scheduler")
        if scheduler_name == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=float(config["train"].get("lr_plateau_factor", 0.5)),
                patience=int(config["train"].get("lr_plateau_patience", 4)),
                min_lr=float(config["train"].get("minimum_learning_rate", 1e-6)),
            )
        elif scheduler_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=(
                    max(
                        epochs
                        * math.ceil(
                            len(self.train_loader)
                            / max(1, int(config["train"].get("gradient_accumulation_steps", 1)))
                        ),
                        1,
                    )
                    if self.scheduler_step_per_optimizer
                    else max(epochs, 1)
                ),
                eta_min=float(config["train"].get("minimum_learning_rate", 1e-6)),
            )
        else:
            raise ValueError("train.scheduler must be cosine or plateau")
        amp_enabled = bool(config["train"].get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled,
            init_scale=float(config["train"].get("amp_init_scale", 65536.0)),
            growth_interval=int(config["train"].get("amp_growth_interval", 2000)),
        )
        self.amp_enabled = amp_enabled
        self.ema: Optional[ModelEMA] = None
        if self.stage in {"private", "private_seg"} or bool(
            config["train"].get("use_ema", False)
        ):
            self.ema = ModelEMA(self.model, decay=float(config["train"].get("ema_decay", 0.999)))
            self.ema.module.to(self.device)
        self.writer = SummaryWriter(self.output_dir / "tensorboard")
        self.csv_logger = CSVLogger(self.output_dir / "history.csv")
        self.start_epoch = 0
        self.best_metric = -math.inf
        self.bad_epochs = 0
        self.phase_machine = (
            PhaseStateMachine(
                str(schedule),
                tuple(int(x) for x in config["train"].get("schedule_epochs", [20, 20, 20])),
                optimizer_steps_per_epoch=math.ceil(
                    len(self.train_loader)
                    / max(1, int(config["train"].get("gradient_accumulation_steps", 1)))
                ),
            )
            if schedule in {"order_ds", "order_sd", "order_alt"} else None
        )
        self._resume_if_needed()
        self._write_initialization_audit()
        self._denoise_probe_image: Optional[torch.Tensor] = None
        self._denoise_probe_reference: Optional[torch.Tensor] = None
        denoising_path_trainable = any(
            parameter.requires_grad
            for module in (
                self.model.adapters["denoise"],
                self.model.decoders["denoise"],
                self.model.residual_head,
            )
            for parameter in module.parameters()
        )
        requested_denoise_drift_monitor = bool(
            config["train"].get("monitor_denoise_drift", False)
        )
        self.monitor_denoise_drift = (
            requested_denoise_drift_monitor and not denoising_path_trainable
        )
        config.setdefault("runtime", {})["effective_monitor_denoise_drift"] = bool(
            self.monitor_denoise_drift
        )
        if requested_denoise_drift_monitor and denoising_path_trainable:
            warnings.warn(
                "monitor_denoise_drift was disabled because the denoising decoder/head "
                "is trainable in this stage; encoder freezing remains independently audited.",
                RuntimeWarning,
            )
        if self.monitor_denoise_drift:
            self._initialize_denoise_probe()

    def _setup_d2_teacher(self) -> None:
        d2_cfg = self.config.get("d2", {})
        if not d2_cfg.get("enabled", False):
            return
        if d2_cfg.get("template_only", False):
            raise ValueError("D2 template is not executable; use tools/prepare_d2_seed42.py")
        if self.config.get("train", {}).get("stage") != "denoise":
            raise ValueError("D2 is only supported with train.stage=denoise")
        if self.config.get("loss", {}).get("restoration_mode") != "structure_d2":
            raise ValueError("D2 requires loss.restoration_mode=structure_d2")
        if self.config.get("train", {}).get("resume"):
            raise ValueError("D2 registered runs must start fresh; resume is not supported")
        if not self.config.get("data", {}).get("load_segmentation_labels", False):
            raise ValueError("D2 requires explicit train/val segmentation label loading")
        teacher_cfg = d2_cfg.get("teacher", {})
        if not teacher_cfg.get("enabled", False):
            if any(float(self.config.get("loss", {}).get("d2", {}).get("weights", {}).get(key, 0.0)) > 0
                   for key in ("teacher_task", "teacher_consistency")):
                raise ValueError("D2 teacher loss is nonzero but no teacher is enabled")
            write_json({
                "status": "not_applicable",
                "enabled": False,
                "reason": "This registered D2 ablation has no teacher objective",
                "requires_grad_parameter_count": 0,
                "changed_parameter_count": 0,
                "test_assets_opened": 0,
            }, self.output_dir / "teacher_audit.json")
            return
        checkpoint_value = teacher_cfg.get("checkpoint")
        expected_sha = teacher_cfg.get("sha256")
        evidence_value = teacher_cfg.get("evidence")
        evidence_sha = teacher_cfg.get("evidence_sha256")
        if not checkpoint_value or not expected_sha or not evidence_value or not evidence_sha:
            raise ValueError("D2 teacher checkpoint, evidence and frozen SHA256 values are required")
        checkpoint_path = Path(checkpoint_value).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing D2 segmentation teacher: {checkpoint_path}")
        actual_sha = _sha256_file(checkpoint_path)
        if actual_sha != expected_sha:
            raise ValueError("D2 segmentation teacher SHA256 mismatch")
        evidence_path = Path(evidence_value).expanduser().resolve()
        if not evidence_path.is_file() or _sha256_file(evidence_path) != evidence_sha:
            raise ValueError("D2 segmentation teacher evidence missing or changed")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8-sig"))
        if (
            d2_cfg.get("run_mode") != "smoke"
            and evidence.get("evidence_type") not in {
                "native_protocol_binding", "derived_legacy_binding"
            }
        ):
            raise ValueError("Formal D2 teacher evidence type is unverified")
        bound_teacher_sha = evidence.get("checkpoint_sha256") or evidence.get("best_checkpoint_sha256")
        if (bound_teacher_sha != actual_sha
                or evidence.get("status", "passed") not in {"passed", "completed"}
                or int(evidence.get("test_assets_opened", -1)) != 0
                or evidence.get("selection_rule") != teacher_cfg.get("selection_rule")
                or evidence.get("training_data") != teacher_cfg.get("training_data")
                or evidence.get("split") != teacher_cfg.get("split")):
            raise ValueError("D2 teacher evidence binds a different checkpoint")
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(raw, dict) or not raw.get("config") or not raw.get("model"):
            raise ValueError("D2 teacher checkpoint lacks model/config provenance")
        teacher_config = raw["config"]
        expected_protocol = self.config.get("runtime", {}).get("active_protocol_lock", {})
        if expected_protocol:
            validate_checkpoint_config(raw, expected_protocol, "D2 segmentation teacher")
        # Teacher presence must not consume the D2 model/data RNG streams and
        # thereby confound paired D20/D24/D25 comparisons.
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
        try:
            teacher = build_model(teacher_config).to(self.device)
            teacher.load_state_dict(raw["model"], strict=True)
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        if any(id(left) == id(right) for left in self.model.parameters() for right in teacher.parameters()):
            raise RuntimeError("D2 and segmentation teacher unexpectedly share parameters")
        from sabids.experiments.dose_response import tensor_sha
        self.d2_teacher = teacher
        self._d2_teacher_initial_sha = {
            name: tensor_sha(parameter) for name, parameter in teacher.named_parameters()
        }
        write_json({
            "status": "passed",
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": actual_sha,
            "selection_rule": teacher_cfg.get("selection_rule"),
            "training_data": teacher_cfg.get("training_data"),
            "split": teacher_cfg.get("split"),
            "evidence": str(evidence_path),
            "evidence_sha256": evidence_sha,
            "parameter_count": sum(parameter.numel() for parameter in teacher.parameters()),
            "requires_grad_parameter_count": sum(
                parameter.numel() for parameter in teacher.parameters() if parameter.requires_grad
            ),
            "shares_parameter_objects_with_d2": False,
            "test_assets_opened": 0,
        }, self.output_dir / "teacher_audit.json")

    def _precompute_d2_teacher_clean_outputs(
        self, batch: Dict[str, torch.Tensor]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Compute the detached clean reference before student graphs exist."""
        if self.d2_teacher is None:
            return None
        self.d2_teacher.eval()
        with torch.no_grad():
            raw = self.d2_teacher(
                batch["clean"], return_features=False, return_auxiliary=False
            )
        clean = {
            "clean_layer_prob": raw["layer_prob"].detach(),
            "clean_vessel_prob": raw["vessel_prob"].detach(),
        }
        del raw
        return clean

    def _attach_d2_teacher_outputs(
        self,
        output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        clean_reference: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        if self.d2_teacher is None:
            return
        self.d2_teacher.eval()
        if clean_reference is None:
            clean_reference = self._precompute_d2_teacher_clean_outputs(batch)
        assert clean_reference is not None

        def teacher_logits(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            prediction = self.d2_teacher(
                image, return_features=False, return_auxiliary=False
            )
            return prediction["layer_logits"], prediction["vessel_logits"]

        if bool(self.config.get("train", {}).get("memory_safe_d2_teacher", False)):
            layer_logits, vessel_logits = checkpoint(
                teacher_logits,
                output["denoised_raw"],
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            layer_logits, vessel_logits = teacher_logits(output["denoised_raw"])
        output.update({
            "d2_teacher_layer_logits": layer_logits,
            "d2_teacher_vessel_logits": vessel_logits,
            "d2_teacher_clean_layer_prob": clean_reference["clean_layer_prob"],
            "d2_teacher_clean_vessel_prob": clean_reference["clean_vessel_prob"],
        })

    def _prepare_formal_d2_teacher_config(self) -> None:
        """Bind the opt-in segmentation teacher to the active protocol.

        This path is deliberately separate from legacy Stage 2.  No behavior
        changes unless ``formal_d2_teacher.enabled`` is explicitly true.
        """
        teacher_cfg = self.config.get("formal_d2_teacher", {})
        if not teacher_cfg.get("enabled", False):
            return
        if teacher_cfg.get("template_only", False):
            raise ValueError(
                "Formal D2 teacher template is not executable; use tools/prepare_d2_teacher.py"
            )
        from sabids.experiments.dose_response import resolve

        if self.config.get("train", {}).get("stage") != "segment":
            raise ValueError("Formal D2 teacher requires train.stage=segment")
        if int(self.config.get("seed", -1)) != 42:
            raise ValueError("The first formal D2 teacher is fixed to seed 42")
        train_cfg = self.config["train"]
        if train_cfg.get("resume"):
            raise ValueError("Formal D2 teacher must start fresh; resume is forbidden")
        if train_cfg.get("monitor") != "vessel_soft_dice":
            raise ValueError("Formal D2 teacher monitor must be vessel_soft_dice")
        if train_cfg.get("checkpoint_selection_rule") != "best_validation_vessel_soft_dice":
            raise ValueError("Formal D2 teacher selection rule mismatch")
        epochs = int(train_cfg.get("epochs", 0))
        if int(train_cfg.get("early_stopping_patience", 0)) <= epochs:
            raise ValueError("Formal D2 teacher must complete its fixed training budget")
        model_cfg = self.config.get("model", {})
        if any(bool(model_cfg.get(key, False)) for key in (
            "d2s_enabled", "s2d_enabled", "enable_denoise_to_seg", "enable_seg_to_denoise",
        )):
            raise ValueError("Formal D2 teacher requires D->S and S->D disabled")
        if not bool(model_cfg.get("stage2_freeze_shared_encoder", False)):
            raise ValueError("Formal D2 teacher requires the safe-current frozen encoder")
        if bool(model_cfg.get("stage2_train_denoise_to_seg", False)):
            raise ValueError("Formal D2 teacher cannot train denoise-to-seg interaction")
        loss = self.config.get("loss", {})
        expected_weights = {
            "layer": 1.0, "vessel": 1.0, "vessel_stroma": 0.25,
            "vessel_area": 0.2, "vessel_outside": 0.0, "containment": 0.1,
        }
        for key, expected in expected_weights.items():
            if float(loss.get("weights", {}).get(key, float("nan"))) != expected:
                raise ValueError(f"Formal D2 teacher safe-current loss mismatch: {key}")
        if float(loss.get("auxiliary_weight", 0.0)) != 0.0:
            raise ValueError("Formal D2 teacher auxiliary loss must be disabled")
        evidence_cfg = self.config.get("training_asset_evidence", {})
        if evidence_cfg.get("enabled") is not True:
            raise ValueError("Formal D2 teacher requires training-time asset evidence")
        project_root = Path(evidence_cfg.get("project_root", ".")).expanduser().resolve()
        lock_path = resolve(project_root, evidence_cfg.get("protocol_lock", ""))
        split_path = resolve(project_root, teacher_cfg.get("split_contract", ""))
        lock = load_protocol_lock(lock_path)
        if not split_path.is_file() or _sha256_file(split_path) != lock["split_contract_sha256"]:
            raise ValueError("Formal D2 teacher split contract is missing or changed")
        if self.config.get("protocol_id") != lock["protocol_id"]:
            raise ValueError("Formal D2 teacher protocol_id differs from active lock")
        if list(self.config["data"].get("target_size", [])) != list(lock["input_resolution"]):
            raise ValueError("Formal D2 teacher target_size differs from active lock")
        if self.config["data"].get("normalization") != lock["normalization"]:
            raise ValueError("Formal D2 teacher normalization differs from active lock")
        protocol_root = resolve(project_root, lock["manifest_root"])
        expected_manifest = (protocol_root / "train_segment.csv").resolve()
        if resolve(project_root, self.config["data"]["manifest"]) != expected_manifest:
            raise ValueError("Formal D2 teacher must use the locked train_segment.csv")
        output = self.output_dir.resolve()
        current_root = (project_root / "runs" / "current").resolve()
        if output == current_root or current_root in output.parents:
            raise ValueError("Formal D2 teacher must not overwrite runs/current")
        for key in ("manifest_root", "data_plan_sha256", "label_inventory_sha256"):
            self.config[key] = lock[key]
        runtime = self.config.setdefault("runtime", {})
        runtime.update({
            "active_protocol_lock": lock,
            "active_protocol_lock_path": str(lock_path),
            "formal_d2_teacher_split_contract": str(split_path),
            "formal_d2_teacher_split_contract_sha256": _sha256_file(split_path),
            "test_assets_opened": 0,
        })

    def _prepare_training_asset_evidence_config(self) -> None:
        evidence_cfg = self.config.get("training_asset_evidence", {})
        if not evidence_cfg.get("enabled", False):
            return
        stage = str(self.config.get("train", {}).get("stage", ""))
        formal_teacher = bool(self.config.get("formal_d2_teacher", {}).get("enabled", False))
        if stage != "denoise" and not (stage == "segment" and formal_teacher):
            raise ValueError(
                "training_asset_evidence requires stage=denoise or an explicit formal D2 teacher"
            )
        if self.config.get("train", {}).get("resume"):
            raise ValueError(
                "training_asset_evidence cannot be enabled retroactively on a resumed run"
            )
        existing_training = [
            path.name
            for path in (
                self.output_dir / "last.pth",
                self.output_dir / "best.pth",
                self.output_dir / "history.csv",
                self.output_dir / "initial.pth",
            )
            if path.exists()
        ]
        if existing_training:
            raise FileExistsError(
                "Refusing retroactive training evidence for existing artifacts: "
                + ", ".join(existing_training)
            )
        from sabids.experiments.dose_response import (
            audit_d1_reproduction_config,
            git_commit,
            resolve,
            write_strict_json_exclusive,
        )
        project_root = Path(evidence_cfg.get("project_root", ".")).expanduser().resolve()
        lock_value = evidence_cfg.get("protocol_lock")
        if lock_value:
            lock_path = resolve(project_root, lock_value)
            lock = load_protocol_lock(lock_path)
            if self.config.get("protocol_id") != lock["protocol_id"]:
                raise ValueError("Training config protocol_id differs from protocol lock")
            bindings = []
            # The launch suite historically injected these values from the
            # active lock into resolved configs. Reproduce that binding before
            # semantic comparison instead of trusting stale template hashes.
            for key in ("manifest_root", "data_plan_sha256", "label_inventory_sha256"):
                previous = self.config.get(key)
                current = lock[key]
                if previous != current:
                    bindings.append({"path": key, "configured": previous, "bound": current})
                self.config[key] = current
            if list(self.config["data"].get("target_size", [])) != list(lock["input_resolution"]):
                raise ValueError("Training target_size differs from protocol lock")
            if self.config["data"].get("normalization") != lock["normalization"]:
                raise ValueError("Training normalization differs from protocol lock")
            runtime = self.config.setdefault("runtime", {})
            runtime["active_protocol_lock"] = lock
            runtime["active_protocol_lock_path"] = str(lock_path)
            runtime["training_asset_protocol_bindings"] = bindings
        if evidence_cfg.get("reference_resolved_config"):
            epochs = int(self.config.get("train", {}).get("epochs", 0))
            if self.config["train"].get("checkpoint_selection_rule") != "fixed_final_primary":
                raise ValueError("Formal D1 reproduction requires fixed_final_primary")
            if int(self.config["train"].get("fixed_epoch", -1)) != epochs:
                raise ValueError("Formal D1 fixed_epoch must equal the configured budget")
            if int(self.config["train"].get("early_stopping_patience", 0)) <= epochs:
                raise ValueError("Formal D1 reproduction must not early-stop before fixed_final")
            audit = audit_d1_reproduction_config(project_root, self.config)
            write_strict_json_exclusive(
                self.output_dir / "d1_reproduction_semantic_audit.json", audit
            )
        self.config.setdefault("runtime", {})["git_commit"] = git_commit(project_root)

    def _record_training_asset_evidence(self) -> None:
        evidence_cfg = self.config.get("training_asset_evidence", {})
        if not evidence_cfg.get("enabled", False):
            return
        from sabids.experiments.dose_response import create_training_asset_evidence
        train = _effective_dataset_table(self.train_loader.dataset)
        val = _effective_dataset_table(self.val_loader.dataset)
        filtered = pd.concat([train, val], ignore_index=True)
        lock = self.config.get("runtime", {}).get("active_protocol_lock")
        if lock:
            train_groups = set(train["group_id"].astype(str).unique())
            val_groups = set(val["group_id"].astype(str).unique())
            d2_run_mode = self.config.get("d2", {}).get("run_mode")
            teacher_protocol = self.config.get("formal_d2_teacher", {})
            teacher_run_mode = teacher_protocol.get("run_mode")
            partial_diagnostic = (
                d2_run_mode in {"smoke", "overfit"}
                or teacher_run_mode in {"smoke", "overfit"}
            )
            if teacher_protocol.get("enabled") is True:
                expected_train = set(map(str, teacher_protocol.get(
                    "expected_train_positions", []
                )))
                expected_val = set(map(str, teacher_protocol.get(
                    "expected_validation_positions", []
                )))
                if not expected_train:
                    raise ValueError("Formal teacher lacks its registered label-eligible train cohort")
                if not expected_train <= set(lock["train_positions"]):
                    raise ValueError("Formal teacher train cohort exceeds the active-lock train cohort")
                if expected_val != set(lock["validation_positions"]):
                    raise ValueError("Formal teacher validation cohort differs from active lock")
                if partial_diagnostic:
                    if not train_groups <= expected_train or not val_groups <= expected_val:
                        raise ValueError("Teacher diagnostic groups exceed the registered teacher cohort")
                    self.config.setdefault("runtime", {})["partial_protocol_diagnostic"] = True
                else:
                    if train_groups != expected_train:
                        raise ValueError("Effective teacher train groups differ from the registered label-eligible cohort")
                    if val_groups != expected_val:
                        raise ValueError("Effective teacher validation groups differ from the registered cohort")
            elif partial_diagnostic:
                if not train_groups <= set(lock["train_positions"]):
                    raise ValueError("D2 diagnostic train groups exceed the locked development train cohort")
                if not val_groups <= set(lock["validation_positions"]):
                    raise ValueError("D2 diagnostic validation groups exceed the locked validation cohort")
                self.config.setdefault("runtime", {})["partial_protocol_diagnostic"] = True
            else:
                if train_groups != set(lock["train_positions"]):
                    raise ValueError("Effective denoising train groups differ from protocol lock")
                if val_groups != set(lock["validation_positions"]):
                    raise ValueError("Effective denoising validation groups differ from protocol lock")
        self._training_asset_filtered = filtered
        self._training_asset_initial_path = (
            self.output_dir / "training_asset_inventory_initial.json"
        )
        project_root = Path(evidence_cfg.get("project_root", ".")).expanduser().resolve()
        evidence = create_training_asset_evidence(
            project_root, self.config, filtered, self._training_asset_initial_path
        )
        runtime = self.config.setdefault("runtime", {})
        if runtime.get("manifest_sha256") != evidence["manifest_sha256"]:
            raise RuntimeError("Training evidence manifest SHA differs from Trainer runtime")
        if runtime.get("effective_split_sha256") != evidence["effective_split_sha256"]:
            raise RuntimeError("Training evidence effective split differs from Trainer runtime")
        runtime["train_val_noisy_clean_asset_sha256"] = evidence[
            "train_val_noisy_clean_asset_sha256"
        ]
        runtime["training_asset_inventory_initial"] = str(
            self._training_asset_initial_path.resolve()
        )

    def _write_initialization_audit(self) -> None:
        """Fingerprint the exact post-load, pre-training state for paired runs."""
        tensor_digests: Dict[str, str] = {}
        aggregate = hashlib.sha256()
        common = hashlib.sha256()
        interaction = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest = hashlib.sha256()
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(value.numpy().tobytes())
            hexdigest = digest.hexdigest()
            tensor_digests[name] = hexdigest
            payload = f"{name}|{hexdigest}\n".encode("utf-8")
            aggregate.update(payload)
            (interaction if name.startswith("interactions.") else common).update(payload)

        optimizer_ids = [
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]
        if len(optimizer_ids) != len(set(optimizer_ids)):
            raise RuntimeError("Optimizer contains duplicate parameter objects")
        sampler_plan = []
        for epoch in range(int(self.config.get("train", {}).get("epochs", 1))):
            self.train_sampler.set_epoch(epoch)
            sampler_plan.append(list(iter(self.train_sampler)))
        self.train_sampler.set_epoch(0)
        plan_payload = json.dumps(
            {
                "sampler_indices": sampler_plan,
                "data_seed": int(self.config.get("seed", 42)) + 1_000_003,
                "num_workers": int(self.config.get("train", {}).get("num_workers", 4)),
                "augmentation": self.config.get("data", {}).get("augmentation", {}),
            },
            sort_keys=True,
        ).encode("utf-8")
        sampler_plan_sha256 = hashlib.sha256(plan_payload).hexdigest()
        write_json(
            {
                "seed": int(self.config.get("seed", 42)),
                "stage": self.stage,
                "model_state_sha256": aggregate.hexdigest(),
                "common_state_sha256": common.hexdigest(),
                "interaction_state_sha256": interaction.hexdigest(),
                "tensor_sha256": tensor_digests,
                "initialization_checkpoint": self.config.get("runtime", {}).get(
                    "initialization_checkpoint"
                ),
                "initialization_checkpoint_sha256": self.config.get("runtime", {}).get(
                    "initialization_checkpoint_sha256"
                ),
                "manifest_sha256": self.config.get("runtime", {}).get("manifest_sha256"),
                "effective_split_sha256": self.config.get("runtime", {}).get(
                    "effective_split_sha256"
                ),
                "label_assets_decoded_sha256": self.config.get("runtime", {}).get(
                    "label_assets_decoded_sha256"
                ),
                "optimizer_parameter_objects": len(optimizer_ids),
                "optimizer_parameter_elements": int(
                    sum(parameter.numel() for group in self.optimizer.param_groups for parameter in group["params"])
                ),
                # This is the seed-dependent sampler/augmentation schedule, not
                # the immutable protocol-level data_plan_sha256.  Keep the old
                # key for consumers of historical factorial audits and expose
                # the unambiguous name for new code.
                "sampler_plan_sha256": sampler_plan_sha256,
                "data_plan_sha256": sampler_plan_sha256,
                "data_plan_sha256_semantics": "legacy_alias_of_sampler_plan_sha256",
                "data_rng_seed": int(self.config.get("seed", 42)) + 1_000_003,
                "model_rng_seed": int(self.config.get("seed", 42)),
            },
            self.output_dir / "initialization_audit.json",
        )
        if self.config.get("formal_d2_teacher", {}).get("enabled", False):
            initial_path = self.output_dir / "initial.pth"
            if initial_path.exists():
                raise FileExistsError(f"Refusing existing formal teacher initial checkpoint: {initial_path}")
            save_checkpoint(
                initial_path,
                self.model,
                self.optimizer,
                self.scheduler,
                -1,
                self.best_metric,
                self.config,
                self.scaler,
                self.ema.state_dict() if self.ema is not None else None,
                {"global_optimizer_step": 0, "formal_d2_teacher_initial": True},
            )
        if self.config.get("dose_response", {}).get("enabled", False):
            from sabids.experiments.dose_response import augmentation_plan_sha, stable_sha, write_strict_json
            audit_path = self.output_dir / "initialization_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            dataset = self.train_loader.dataset
            ids = dataset.table.sample_id.astype(str).tolist()
            audit["actual_augmentation_plan_sha256"] = augmentation_plan_sha(
                int(self.config["seed"]), int(self.config["train"]["epochs"]), ids,
                float(self.config["data"]["augmentation"]["horizontal_flip"]))
            audit["augmentation_algorithm"] = "sha256(seed,epoch,sample_id); horizontal flip only"
            audit["trainable_parameter_names_sha256"] = stable_sha(
                [n for n, p in self.model.named_parameters() if p.requires_grad])
            audit["paired_cohort_sha256"] = stable_sha({
                "train": dataset.table[["sample_id", "group_id", "split"]].to_dict("records"),
                "val": self.val_loader.dataset.dataset.table[["sample_id", "group_id", "split"]].to_dict("records")
                    if isinstance(self.val_loader.dataset, Subset) else
                    self.val_loader.dataset.table[["sample_id", "group_id", "split"]].to_dict("records")})
            write_strict_json(audit_path, audit)
            write_strict_json(self.output_dir / "data_plan.json", {
                "paired_cohort_sha256": audit["paired_cohort_sha256"],
                "sampler_plan_sha256": audit["sampler_plan_sha256"],
                "actual_augmentation_plan_sha256": audit["actual_augmentation_plan_sha256"],
                "train_sample_ids": ids, "sampler_indices_by_epoch": sampler_plan,
                "dose_response": self.config["dose_response"], "test_assets_opened": 0})

    def _record_run_inputs(self) -> None:
        runtime = self.config.setdefault("runtime", {})
        manifest = Path(self.config["data"]["manifest"]).expanduser().resolve()
        runtime["manifest_sha256"] = _sha256_file(manifest)
        table = pd.read_csv(manifest, dtype=str).fillna("")
        runtime["group_ids_by_split"] = {
            str(split): sorted(part["group_id"].astype(str).unique().tolist())
            for split, part in table.groupby("split")
        }
        runtime["rows_by_split"] = {
            str(key): int(value)
            for key, value in table["split"].value_counts().items()
        }
        label_assets = []
        load_segmentation_labels = _load_segmentation_labels(self.config)
        runtime["load_segmentation_labels"] = load_segmentation_labels
        asset_table = table if load_segmentation_labels else table.iloc[0:0]
        if not load_segmentation_labels:
            runtime["label_inventory_splits"] = []
        elif (
            str(self.config.get("train", {}).get("stage", "")) in {"interaction", "input_segment"}
            or bool(self.config.get("formal_d2_teacher", {}).get("enabled", False))
        ):
            allowed_splits = {
                str(self.config.get("data", {}).get("train_split", "train")),
                str(self.config.get("data", {}).get("val_split", "val")),
            }
            asset_table = table[table["split"].astype(str).isin(allowed_splits)]
            runtime["label_inventory_splits"] = sorted(allowed_splits)
        data_root = self.config.get("data", {}).get("root")
        root = Path(data_root).expanduser().resolve() if data_root else manifest.parent
        for column in (
            "layer_mask_path",
            "vessel_mask_path",
            "label_valid_mask_path",
            "vessel_valid_mask_path",
            "multiclass_label_path",
        ):
            if column not in asset_table.columns:
                continue
            logical_assets = (
                asset_table.loc[asset_table[column].astype(str) != "", ["group_id", column]]
                .drop_duplicates()
                .sort_values(["group_id", column])
            )
            group_ordinals: Dict[str, int] = defaultdict(int)
            for _, asset_row in logical_assets.iterrows():
                value = str(asset_row[column])
                group_id = str(asset_row["group_id"])
                ordinal = group_ordinals[group_id]
                group_ordinals[group_id] += 1
                asset = Path(value).expanduser()
                if not asset.is_absolute():
                    asset = (root / asset).resolve()
                inspection = _inspect_label_asset(asset, allow_float_cache=bool(
                    self.config.get("dose_response", {}).get("enabled", False))) if asset.is_file() else {}
                label_assets.append(
                    {
                        "asset_id": f"{group_id}|{column}|{ordinal}",
                        "group_id": group_id,
                        "column": column,
                        "path": str(asset),
                        **inspection,
                    }
                )
        raw_payload = "\n".join(
            f"{item['asset_id']}|{item.get('raw_sha256')}"
            for item in label_assets
        ).encode("utf-8")
        decoded_payload = "\n".join(
            f"{item['asset_id']}|{item.get('decoded_sha256')}"
            for item in label_assets
        ).encode("utf-8")
        runtime["hash_schema_version"] = "stage2-fingerprint-v2"
        runtime["metadata_version"] = 2
        runtime["label_assets_raw_sha256"] = hashlib.sha256(
            raw_payload
        ).hexdigest()
        runtime["label_assets_decoded_sha256"] = hashlib.sha256(
            decoded_payload
        ).hexdigest()
        runtime["label_assets_sha256"] = runtime["label_assets_raw_sha256"]
        runtime["label_asset_count"] = len(label_assets)
        runtime["missing_label_assets"] = [
            item["path"] for item in label_assets if item.get("raw_sha256") is None
        ]
        write_json(label_assets, self.output_dir / "label_asset_inventory.json")
        data_config = self.config.get("data", {})
        runtime["effective_groups"] = {}
        runtime["effective_rows"] = {}
        for role in ("train", "val"):
            if self.config.get("training_asset_evidence", {}).get("enabled", False):
                loader = self.train_loader if role == "train" else self.val_loader
                part = _effective_dataset_table(loader.dataset)
            else:
                split = str(data_config.get(f"{role}_split", role))
                part = table[table["split"].astype(str) == split]
                configured_groups = data_config.get(f"{role}_groups")
                if configured_groups:
                    allowed = {str(value) for value in configured_groups}
                    part = part[part["group_id"].astype(str).isin(allowed)]
            runtime["effective_groups"][role] = sorted(
                part["group_id"].astype(str).unique().tolist()
            )
            runtime["effective_rows"][role] = int(len(part))
        split_payload = "\n".join(
            f"{role}:{group_id}"
            for role in ("train", "val")
            for group_id in runtime["effective_groups"][role]
        ).encode("utf-8")
        runtime["effective_split_sha256"] = hashlib.sha256(
            split_payload
        ).hexdigest()
        pretrained = self.config["train"].get("pretrained") or runtime.get(
            "pretrained_source"
        )
        if pretrained:
            checkpoint = Path(pretrained).expanduser().resolve()
            runtime["initialization_checkpoint"] = str(checkpoint)
            if checkpoint.is_file():
                stat = checkpoint.stat()
                runtime["initialization_checkpoint_size"] = int(stat.st_size)
                runtime["initialization_checkpoint_mtime_ns"] = int(
                    stat.st_mtime_ns
                )
                runtime["initialization_checkpoint_sha256"] = _sha256_file(
                    checkpoint
                )

    def _write_run_metadata(
        self, best_epoch: int, monitor: str, best_checkpoint: Path
    ) -> None:
        evaluation = self.config.get("evaluation", {})
        metadata = {
            "git_commit": self.config.get("runtime", {}).get("git_commit"),
            "metadata_version": self.config.get("runtime", {}).get(
                "metadata_version"
            ),
            "hash_schema_version": self.config.get("runtime", {}).get(
                "hash_schema_version"
            ),
            "manifest_sha256": self.config.get("runtime", {}).get(
                "manifest_sha256"
            ),
            "effective_split_sha256": self.config.get("runtime", {}).get(
                "effective_split_sha256"
            ),
            "effective_groups": self.config.get("runtime", {}).get(
                "effective_groups", {}
            ),
            "group_ids_by_split": self.config.get("runtime", {}).get(
                "group_ids_by_split", {}
            ),
            "rows_by_split": self.config.get("runtime", {}).get(
                "rows_by_split", {}
            ),
            "label_assets_sha256": self.config.get("runtime", {}).get(
                "label_assets_sha256"
            ),
            "label_assets_raw_sha256": self.config.get("runtime", {}).get(
                "label_assets_raw_sha256"
            ),
            "label_assets_decoded_sha256": self.config.get("runtime", {}).get(
                "label_assets_decoded_sha256"
            ),
            "label_asset_inventory": str(
                (self.output_dir / "label_asset_inventory.json").resolve()
            ),
            "label_asset_count": self.config.get("runtime", {}).get(
                "label_asset_count"
            ),
            "missing_label_assets": self.config.get("runtime", {}).get(
                "missing_label_assets", []
            ),
            "initialization_checkpoint": self.config.get("runtime", {}).get(
                "initialization_checkpoint"
            ),
            "initialization_checkpoint_sha256": self.config.get(
                "runtime", {}
            ).get("initialization_checkpoint_sha256"),
            "best_epoch": int(best_epoch),
            "monitor": monitor,
            "best_metric": float(self.best_metric),
            "layer_threshold": float(
                evaluation.get("layer_threshold", evaluation.get("threshold", 0.5))
            ),
            "vessel_threshold": float(
                evaluation.get("vessel_threshold", evaluation.get("threshold", 0.5))
            ),
            "best_checkpoint": str(best_checkpoint.resolve()),
            "best_checkpoint_sha256": (
                _sha256_file(best_checkpoint) if best_checkpoint.is_file() else None
            ),
        }
        if (self.config.get("training_asset_evidence", {}).get("enabled", False)
                or self.config.get("d2", {}).get("enabled", False)):
            metadata.update({
                "run_id": self.output_dir.name,
                "selection_rule": f"best_validation_{monitor}",
            })
        write_json(metadata, self.output_dir / "run_metadata.json")

    @torch.no_grad()
    def _initialize_denoise_probe(self) -> None:
        batch = next(iter(self.val_loader))
        self._denoise_probe_image = batch["image"][:1].to(self.device)
        self.model.eval()
        self._denoise_probe_reference = self.model(
            self._denoise_probe_image,
            return_features=False,
            return_auxiliary=False,
        )["denoised"].detach().clone()

    def _resume_if_needed(self) -> None:
        resume = self.config["train"].get("resume")
        pretrained = self.config["train"].get("pretrained")
        expected = self.config.get("runtime", {}).get("active_protocol_lock", {})

        def validate_protocol(checkpoint_path: str, checkpoint: Dict) -> None:
            effective = {
                key: expected.get(key, self.config.get(key))
                for key in ("protocol_id", "data_plan_sha256", "label_inventory_sha256")
            }
            validate_checkpoint_config(checkpoint, effective, checkpoint_path)

        if resume:
            raw = torch.load(resume, map_location="cpu", weights_only=False)
            validate_protocol(str(resume), raw)
            checkpoint = load_checkpoint(
                resume,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                strict=True,
                map_location=self.device,
            )
            self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
            self.best_metric = float(checkpoint.get("best_metric", -math.inf))
            phase_state = checkpoint.get("phase_state") or checkpoint.get("config", {}).get("runtime", {}).get("phase_state")
            if self.phase_machine is not None and phase_state:
                self.phase_machine.restore(phase_state)
                batch_plan_state = phase_state.get("batch_plan_state", {})
                if batch_plan_state.get("train_loader_generator_state") is not None and self.train_loader.generator is not None:
                    self.train_loader.generator.set_state(batch_plan_state["train_loader_generator_state"])
                if batch_plan_state.get("val_loader_generator_state") is not None and self.val_loader.generator is not None:
                    self.val_loader.generator.set_state(batch_plan_state["val_loader_generator_state"])
            if self.ema is not None:
                if checkpoint.get("ema") is not None:
                    self.ema.load_state_dict(checkpoint["ema"])
                else:
                    # Older checkpoints may not contain an EMA state. Start the
                    # teacher from the restored student instead of a random copy.
                    self.ema.load_state_dict(self.model.state_dict())
        elif pretrained:
            raw = torch.load(pretrained, map_location="cpu", weights_only=False)
            validate_protocol(str(pretrained), raw)
            load_checkpoint(
                pretrained,
                self.model,
                strict=bool(self.config["train"].get("strict_pretrained", False)),
                map_location=self.device,
            )
            if bool(self.config.get("model", {}).get("reset_interaction_scales_after_pretrained", False)):
                value = float(self.config.get("model", {}).get("interaction_scale_init", 0.0))
                with torch.no_grad():
                    for name, parameter in self.model.named_parameters():
                        if name.endswith(("seg_scale", "layer_scale", "vessel_scale")):
                            parameter.fill_(value)
            if self.ema is not None:
                # ModelEMA is constructed before checkpoint loading. Synchronize
                # it here so private pseudo-labels and validation start from the
                # public joint model, not from the random initialization.
                self.ema.load_state_dict(self.model.state_dict())

    @staticmethod
    def _select_output(
        output: Dict[str, torch.Tensor], keys: tuple[str, ...]
    ) -> Dict[str, torch.Tensor]:
        return {key: output[key] for key in keys if key in output}

    def _forward_auxiliary(
        self,
        batch: Dict[str, torch.Tensor],
        detach_cross: bool,
    ) -> tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
        repeat_output = None
        clean_output = None
        teacher_output = None
        train_cfg = self.config["train"]
        loss_weights = self.config.get("loss", {}).get("weights", {})
        rmac_active = float(loss_weights.get("rmac", 0.0)) > 0.0
        identity_active = float(loss_weights.get("identity", 0.0)) > 0.0
        memory_safe = bool(train_cfg.get("memory_safe_joint", True))
        stopgrad_repeat = bool(
            train_cfg.get("stopgrad_repeat_teacher", memory_safe)
        )

        if (
            self.stage in {"joint", "private"}
            and rmac_active
            and bool(batch["has_repeat"].any())
        ):
            repeat_context = torch.no_grad() if stopgrad_repeat else nullcontext()
            with repeat_context:
                raw_repeat = self.model(
                    batch["repeat"],
                    detach_cross=detach_cross,
                    return_features=True,
                    return_auxiliary=False,
                )
            repeat_output = self._select_output(
                raw_repeat,
                (
                    "denoised_raw",
                    "layer_prob",
                    "vessel_prob",
                    "anatomy_embedding",
                ),
            )
            del raw_repeat

        identity_valid = batch["has_clean"].bool() | batch["is_clean"].bool()
        needs_clean_teacher = (
            self.stage in {"joint", "private"}
            and rmac_active
            and bool((batch["has_repeat"].bool() & batch["has_clean"].bool()).any())
        )
        needs_identity = (
            self.stage in {"denoise", "warmup", "joint", "private"}
            and identity_active
            and bool(identity_valid.any())
        )
        if needs_clean_teacher or needs_identity:
            identity_input = torch.where(
                batch["has_clean"].view(-1, 1, 1, 1),
                batch["clean"],
                batch["image_weak"],
            )
            clean_no_grad = needs_clean_teacher and bool(
                train_cfg.get("clean_teacher_no_grad", memory_safe)
            )
            clean_context = torch.no_grad() if clean_no_grad else nullcontext()
            with clean_context:
                raw_clean = (
                    self.model.forward_denoise_only(identity_input)
                    if self.stage == "denoise"
                    else self.model(
                        identity_input,
                        detach_cross=detach_cross,
                        return_features=False,
                        return_auxiliary=False,
                    )
                )
            clean_keys = ["layer_prob", "vessel_prob"]
            if needs_identity:
                clean_keys.append("denoised_raw")
            clean_output = self._select_output(raw_clean, tuple(clean_keys))
            del raw_clean

        if self.ema is not None and self.stage in {"private", "private_seg"}:
            with torch.no_grad():
                raw_teacher = self.ema.module(
                    batch["image_weak"],
                    return_features=False,
                    return_auxiliary=False,
                )
            teacher_output = self._select_output(
                raw_teacher, ("layer_prob", "vessel_prob")
            )
            del raw_teacher
        return repeat_output, clean_output, teacher_output

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.model.enforce_frozen_eval()
        if self.d2_teacher is not None:
            self.d2_teacher.eval()
        if self.phase_machine is not None:
            self.phase_machine.record_epoch_phase(epoch)
            PhaseStateMachine.set_trainable(self.model, self.phase_machine.phase(epoch))
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.train_sampler.set_epoch(epoch)
        if self.config.get("dose_response", {}).get("enabled", False):
            self.train_loader.dataset.set_epoch(epoch)
        totals = defaultdict(float)
        steps = 0
        optimizer_steps = 0
        gradient_norm_total = 0.0
        seen_groups = set()
        layer_supervised_samples = 0
        vessel_supervised_samples = 0
        shared_encoder_start = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if name.startswith(("stem", "encoder_blocks", "downsamples"))
        }
        d2s_scale_start = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
            and (name.endswith("layer_scale") or name.endswith("vessel_scale"))
        }
        s2d_scale_start = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and name.endswith("seg_scale")
        }
        d2s_mapping_start = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "interactions" in name
            and any(token in name for token in (
                "noise_head", "restoration_context", "denoise_to_layer", "denoise_to_vessel"
            )) and not name.endswith(("layer_scale", "vessel_scale"))
        }
        s2d_mapping_start = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "interactions" in name
            and any(token in name for token in ("layer_anatomy", "vessel_anatomy", "seg_to_denoise_gate"))
        }
        d2s_gradient_norm_total = 0.0
        s2d_gradient_norm_total = 0.0
        d2s_scale_gradient_total = 0.0
        gradient_group_totals = defaultdict(float)
        gradient_group_fraction_totals = defaultdict(float)
        scale_gradient_totals = defaultdict(float)
        level_mapping_gradient_totals = defaultdict(float)
        clipping_coefficient_total = 0.0
        post_clip_gradient_norm_total = 0.0
        amp_overflow_skips = 0
        consecutive_amp_overflows = 0
        minimum_grad_scale = float(self.scaler.get_scale())
        diagnostic_counts = defaultdict(int)
        shared_gradient_cosines = []
        shared_denoise_gradient_norms = []
        shared_segmentation_gradient_norms = []
        accumulation_steps = max(
            1, int(self.config["train"].get("gradient_accumulation_steps", 1))
        )
        detach_epochs = int(self.config["train"].get("detach_cross_epochs", 10))
        detach_cross = epoch < detach_epochs
        ramp_epochs = max(int(self.config["train"].get("ramp_epochs", 20)), 1)
        ramp = min(1.0, max(0.0, (epoch + 1) / ramp_epochs))
        interaction_ramp_epochs = max(
            1, int(self.config["train"].get("interaction_ramp_epochs", ramp_epochs))
        )
        self.model.set_interaction_progress((epoch + 1) / interaction_ramp_epochs)
        progress = tqdm(self.train_loader, desc=f"Train {epoch + 1}", leave=False)
        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(progress):
            active_phase = self.phase_machine.phase(epoch, batch_index) if self.phase_machine is not None else self.stage
            if self.phase_machine is not None and self.phase_machine.schedule == "order_alt":
                PhaseStateMachine.set_trainable(self.model, active_phase)
            batch = {
                key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            seen_groups.update(str(value) for value in batch["group_id"])
            layer_supervised_samples += int(batch["has_layer"].sum().item())
            vessel_supervised_samples += int(batch["has_vessel"].sum().item())
            amp_context = (
                torch.cuda.amp.autocast() if self.amp_enabled else nullcontext()
            )
            with amp_context:
                # The clean teacher target is detached.  Compute it before any
                # student graph so its transient activations never overlap the
                # clean/noisy student and differentiable teacher graphs.
                d2_teacher_clean = self._precompute_d2_teacher_clean_outputs(batch)
                repeat_output, clean_output, teacher_output = self._forward_auxiliary(
                    batch, detach_cross
                )
                output = (
                    self.model.forward_denoise_only(batch["image"])
                    if active_phase == "denoise"
                    else self.model(
                        batch["image"],
                        detach_cross=detach_cross,
                        interaction_guidance_image=batch.get("interaction_guidance"),
                        **({"return_auxiliary": False, "return_features": False}
                           if self.config.get("dose_response", {}).get("enabled", False) else {}),
                    )
                )
                self._attach_d2_teacher_outputs(output, batch, d2_teacher_clean)
                losses = self.loss_fn(
                    output,
                    batch,
                    stage=active_phase,
                    repeat_output=repeat_output,
                    clean_output=clean_output,
                    teacher_output=teacher_output,
                    ramp=ramp,
                )
                if self.phase_machine is not None and active_phase == "joint" and batch_index == 0:
                    shared_parameters = [
                        parameter
                        for name, parameter in self.model.named_parameters()
                        if parameter.requires_grad
                        and name.startswith(("stem", "encoder_blocks", "downsamples"))
                    ]
                    denoise_objective = losses["reconstruction_weighted"] + losses["residual_weighted"]
                    segmentation_objective = (
                        losses["layer_weighted"]
                        + losses["vessel_weighted"]
                        + losses["vessel_outside_weighted"]
                        + losses["containment_weighted"]
                    )
                    d_grad = torch.autograd.grad(
                        denoise_objective, shared_parameters, retain_graph=True, allow_unused=True
                    )
                    s_grad = torch.autograd.grad(
                        segmentation_objective, shared_parameters, retain_graph=True, allow_unused=True
                    )
                    d_vector = torch.cat([value.reshape(-1) for value in d_grad if value is not None])
                    s_vector = torch.cat([value.reshape(-1) for value in s_grad if value is not None])
                    if d_vector.numel() and s_vector.numel():
                        d_norm = torch.linalg.vector_norm(d_vector)
                        s_norm = torch.linalg.vector_norm(s_vector)
                        cosine = torch.dot(d_vector, s_vector) / (d_norm * s_norm + 1e-12)
                        shared_denoise_gradient_norms.append(float(d_norm.item()))
                        shared_segmentation_gradient_norms.append(float(s_norm.item()))
                        shared_gradient_cosines.append(float(cosine.item()))
            if not bool(torch.isfinite(losses["total"])):
                components = {
                    key: float(value.detach().float().item())
                    for key, value in losses.items()
                    if torch.is_tensor(value) and value.numel() == 1
                }
                raise FloatingPointError(
                    "Non-finite training loss before backward: "
                    f"epoch={epoch + 1}, batch={batch_index + 1}, "
                    f"samples={batch.get('sample_id')}, components={components}"
                )
            accumulation_group_start = (
                batch_index // accumulation_steps
            ) * accumulation_steps
            accumulation_group_size = min(
                accumulation_steps,
                len(self.train_loader) - accumulation_group_start,
            )
            backward_loss = losses["total"] / accumulation_group_size
            self.scaler.scale(backward_loss).backward()
            should_step = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == len(self.train_loader)
            )
            if should_step:
                self.scaler.unscale_(self.optimizer)
                nonfinite_gradient_parameters = [
                    name for name, parameter in self.model.named_parameters()
                    if parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                ]
                if nonfinite_gradient_parameters:
                    if not self.amp_enabled:
                        raise FloatingPointError(
                            f"Non-finite gradients without AMP at epoch={epoch + 1}, "
                            f"batch={batch_index + 1}, samples={batch.get('sample_id')}, "
                            f"parameters={nonfinite_gradient_parameters[:5]}"
                        )
                    # GradScaler has already recorded found_inf in unscale_().
                    # scaler.step() therefore skips the optimizer mutation and
                    # scaler.update() lowers the dynamic scale.
                    previous_scale = float(self.scaler.get_scale())
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    current_scale = float(self.scaler.get_scale())
                    minimum_grad_scale = min(minimum_grad_scale, current_scale)
                    self.optimizer.zero_grad(set_to_none=True)
                    amp_overflow_skips += 1
                    consecutive_amp_overflows += 1
                    maximum = int(self.config["train"].get("max_consecutive_amp_overflows", 8))
                    if consecutive_amp_overflows > maximum:
                        raise FloatingPointError(
                            "Persistent AMP gradient overflow: "
                            f"epoch={epoch + 1}, batch={batch_index + 1}, "
                            f"samples={batch.get('sample_id')}, previous_scale={previous_scale}, "
                            f"current_scale={current_scale}, parameters={nonfinite_gradient_parameters[:5]}"
                        )
                    steps += 1
                    for key, value in losses.items():
                        scalar = float(value.detach().item()) if torch.is_tensor(value) else float(value)
                        totals[key] += scalar
                    totals["amp_overflow_previous_scale"] += previous_scale
                    totals["amp_overflow_current_scale"] += current_scale
                    progress.set_postfix(
                        loss=totals["total"] / steps,
                        amp_overflow=amp_overflow_skips,
                        grad_scale=current_scale,
                    )
                    del output, repeat_output, clean_output, teacher_output, losses
                    del d2_teacher_clean
                    continue
                consecutive_amp_overflows = 0
                d2s_gradients = [
                    parameter.grad.detach().float().reshape(-1)
                    for name, parameter in self.model.named_parameters()
                    if parameter.grad is not None
                    and "interactions" in name
                    and any(
                        token in name
                        for token in (
                            "noise_head",
                            "restoration_context",
                            "denoise_to_layer",
                            "denoise_to_vessel",
                            "layer_scale",
                            "vessel_scale",
                        )
                    )
                ]
                if d2s_gradients:
                    d2s_gradient_norm_total += float(
                        torch.linalg.vector_norm(torch.cat(d2s_gradients)).item()
                    )
                s2d_gradients = [
                    parameter.grad.detach().float().reshape(-1)
                    for name, parameter in self.model.named_parameters()
                    if parameter.grad is not None and "interactions" in name
                    and any(token in name for token in (
                        "layer_anatomy", "vessel_anatomy", "seg_to_denoise_gate", "seg_scale"
                    ))
                ]
                if s2d_gradients:
                    s2d_gradient_norm_total += float(
                        torch.linalg.vector_norm(torch.cat(s2d_gradients)).item()
                    )
                scale_gradients = [
                    parameter.grad.detach().float().abs().mean()
                    for name, parameter in self.model.named_parameters()
                    if parameter.grad is not None
                    and (name.endswith("layer_scale") or name.endswith("vessel_scale"))
                ]
                if scale_gradients:
                    d2s_scale_gradient_total += float(
                        torch.stack(scale_gradients).mean().item()
                    )
                level_mapping_gradients = defaultdict(list)
                for name, parameter in self.model.named_parameters():
                    if parameter.grad is None or not name.startswith("interactions."):
                        continue
                    parts = name.split(".")
                    if len(parts) < 3:
                        continue
                    level, local_name = parts[1], ".".join(parts[2:])
                    if local_name in {"seg_scale", "layer_scale", "vessel_scale"}:
                        scale_gradient_totals[(level, local_name)] += float(
                            parameter.grad.detach().float().abs().mean().item()
                        )
                        continue
                    direction = "d2s" if any(token in local_name for token in (
                        "noise_head", "restoration_context", "denoise_to_layer", "denoise_to_vessel",
                    )) else "s2d"
                    level_mapping_gradients[(level, direction)].append(
                        parameter.grad.detach().float().reshape(-1)
                    )
                for key, values in level_mapping_gradients.items():
                    level_mapping_gradient_totals[key] += float(
                        torch.linalg.vector_norm(torch.cat(values)).item()
                    )
                grouped_gradients = defaultdict(list)
                for name, parameter in self.model.named_parameters():
                    if parameter.grad is None:
                        continue
                    if "interactions" in name and any(token in name for token in (
                        "noise_head", "restoration_context", "denoise_to_layer", "denoise_to_vessel",
                        "layer_scale", "vessel_scale",
                    )):
                        group = "d2s"
                    elif "interactions" in name:
                        group = "s2d"
                    elif name.startswith(("stem", "encoder_blocks", "downsamples")):
                        group = "shared_encoder"
                    elif ".denoise" in name or name.startswith("residual_head"):
                        group = "denoise"
                    elif ".layer" in name or name.startswith(("layer_head", "boundary_head")):
                        group = "layer"
                    elif ".vessel" in name or name.startswith("vessel_head"):
                        group = "vessel"
                    else:
                        group = "other"
                    grouped_gradients[group].append(parameter.grad.detach().float().reshape(-1))
                group_norms = {
                    group: float(torch.linalg.vector_norm(torch.cat(values)).item())
                    for group, values in grouped_gradients.items() if values
                }
                for group, value in group_norms.items():
                    gradient_group_totals[group] += value
                squared_norm_sum = sum(value * value for value in group_norms.values())
                for group, value in group_norms.items():
                    gradient_group_fraction_totals[group] += (
                        value * value / max(squared_norm_sum, 1e-24)
                    )
                gradient_norm = clip_grad_norm_(
                    self.model.parameters(),
                    float(self.config["train"].get("gradient_clip", 1.0)),
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise FloatingPointError(
                        f"Non-finite gradient norm at epoch={epoch + 1}, "
                        f"batch={batch_index + 1}, samples={batch.get('sample_id')}"
                    )
                gradient_norm_total += float(gradient_norm.detach().item())
                clip_limit = float(self.config["train"].get("gradient_clip", 1.0))
                coefficient = min(1.0, clip_limit / (float(gradient_norm.detach().item()) + 1e-12))
                clipping_coefficient_total += coefficient
                post_clip_gradient_norm_total += float(gradient_norm.detach().item()) * coefficient
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler_step_per_optimizer:
                    self.scheduler.step()
                minimum_grad_scale = min(minimum_grad_scale, float(self.scaler.get_scale()))
                optimizer_steps += 1
                if self.phase_machine is not None:
                    self.phase_machine.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.model)
            steps += 1
            for key, value in losses.items():
                scalar = float(value.detach().item()) if torch.is_tensor(value) else float(value)
                totals[key] += scalar
            auxiliary = output.get("auxiliary", [])
            if auxiliary:
                for name in (
                    "seg_to_denoise_injection_relative_rms",
                    "denoise_to_layer_injection_abs_mean",
                    "denoise_to_vessel_injection_abs_mean",
                    "denoise_to_layer_injection_relative_rms",
                    "denoise_to_vessel_injection_relative_rms",
                    "requested_rho",
                    "actual_rho_mean",
                    "actual_rho_median",
                    "actual_rho_p95",
                    "actual_rho_max",
                    "delta_rms",
                    "layer_scale_abs_mean",
                    "vessel_scale_abs_mean",
                    "seg_scale_abs_mean",
                    "guidance_layer_probability_mean",
                    "guidance_vessel_probability_mean",
                    "guidance_layer_probability_std",
                    "guidance_vessel_probability_std",
                    "guidance_layer_probability_min",
                    "guidance_layer_probability_max",
                    "guidance_vessel_probability_min",
                    "guidance_vessel_probability_max",
                    "guidance_layer_confidence_mean",
                    "guidance_layer_confidence_std",
                    "guidance_vessel_confidence_mean",
                    "guidance_vessel_confidence_std",
                    "guidance_finite",
                    "denoise_guidance_mean",
                    "denoise_guidance_std",
                    "denoise_guidance_finite",
                    "seg_scale_signed_mean",
                    "layer_scale_signed_mean",
                    "vessel_scale_signed_mean",
                    "seg_source_layer_rms",
                    "seg_source_vessel_rms",
                    "seg_transformed_anatomy_rms",
                    "denoise_receiver_rms",
                    "denoise_source_rms",
                    "restoration_transformed_rms",
                    "layer_receiver_rms",
                    "vessel_receiver_rms",
                    "s2d_gate_mean", "s2d_gate_std", "s2d_gate_min", "s2d_gate_max",
                    "s2d_gate_saturation_fraction", "s2d_gate_entropy",
                    "d2l_gate_mean", "d2l_gate_std", "d2l_gate_min", "d2l_gate_max",
                    "d2l_gate_saturation_fraction", "d2l_gate_entropy",
                    "d2v_gate_mean", "d2v_gate_std", "d2v_gate_min", "d2v_gate_max",
                    "d2v_gate_saturation_fraction", "d2v_gate_entropy",
                ):
                    values = [float(item[name].item()) for item in auxiliary if name in item]
                    if values:
                        totals[f"interaction_{name}"] += float(np.mean(values))
                        if len(batch.get("dataset", [])) == 1:
                            dataset_name = str(batch["dataset"][0]).replace(" ", "_")
                            labelled = "vessel_labelled" if bool(batch["has_vessel"][0]) else "vessel_unlabelled"
                            totals[f"interaction_dataset_{dataset_name}_{name}"] += float(np.mean(values))
                            totals[f"interaction_{labelled}_{name}"] += float(np.mean(values))
                            diagnostic_counts[f"interaction_dataset_{dataset_name}_{name}"] += 1
                            diagnostic_counts[f"interaction_{labelled}_{name}"] += 1
                    for item in auxiliary:
                        if name in item and "level" in item:
                            level = int(item["level"].item())
                            totals[
                                f"interaction_level{level}_{name}"
                            ] += float(item[name].item())
            progress.set_postfix(loss=totals["total"] / steps)
            del output, repeat_output, clean_output, teacher_output, losses
            del d2_teacher_clean
        result = {key: value / max(steps, 1) for key, value in totals.items()}
        if self.d2_teacher is not None:
            from sabids.experiments.dose_response import tensor_sha
            changed = [
                name for name, parameter in self.d2_teacher.named_parameters()
                if tensor_sha(parameter) != self._d2_teacher_initial_sha[name]
            ]
            result["teacher_changed_parameter_count"] = float(len(changed))
            result["teacher_requires_grad_parameter_count"] = float(sum(
                parameter.numel() for parameter in self.d2_teacher.parameters()
                if parameter.requires_grad
            ))
            if changed or result["teacher_requires_grad_parameter_count"]:
                raise RuntimeError(f"Frozen D2 teacher audit failed: {changed[:5]}")
        for key, count in diagnostic_counts.items():
            result[key] = totals[key] / max(count, 1)
        result.update(
            {
                "optimizer_steps": float(optimizer_steps),
                "gradient_norm": gradient_norm_total / max(optimizer_steps, 1),
                "unique_groups_seen": float(len(seen_groups)),
                "layer_supervised_samples": float(layer_supervised_samples),
                "vessel_supervised_samples": float(vessel_supervised_samples),
                "d2s_gradient_norm": d2s_gradient_norm_total
                / max(optimizer_steps, 1),
                "s2d_gradient_norm": s2d_gradient_norm_total
                / max(optimizer_steps, 1),
                "d2s_scale_gradient_abs_mean": d2s_scale_gradient_total
                / max(optimizer_steps, 1),
                "gradient_norm_after_clip": post_clip_gradient_norm_total / max(optimizer_steps, 1),
                "gradient_clip_coefficient": clipping_coefficient_total / max(optimizer_steps, 1),
                "interaction_scale_weight_decay": float(self.config["train"].get("weight_decay", 0.0)),
                "amp_overflow_skips": float(amp_overflow_skips),
                "amp_minimum_grad_scale": float(minimum_grad_scale),
                "phase_global_step": float(
                    self.phase_machine.global_step if self.phase_machine is not None else 0
                ),
                "shared_denoise_gradient_norm": float(np.mean(shared_denoise_gradient_norms)) if shared_denoise_gradient_norms else 0.0,
                "shared_segmentation_gradient_norm": float(np.mean(shared_segmentation_gradient_norms)) if shared_segmentation_gradient_norms else 0.0,
                "shared_gradient_cosine": float(np.mean(shared_gradient_cosines)) if shared_gradient_cosines else 0.0,
                "shared_negative_cosine_fraction": float(np.mean(np.asarray(shared_gradient_cosines) < 0.0)) if shared_gradient_cosines else 0.0,
                "shared_gradient_cosine_available": float(bool(shared_gradient_cosines)),
            }
        )
        for group, value in gradient_group_totals.items():
            result[f"gradient_group_{group}_norm"] = value / max(optimizer_steps, 1)
            result[f"gradient_group_{group}_fraction"] = (
                gradient_group_fraction_totals[group] / max(optimizer_steps, 1)
            )
        if self.device.type == "cuda":
            result["cuda_peak_memory_bytes"] = float(
                torch.cuda.max_memory_allocated(self.device)
            )
        if d2s_scale_start:
            named_parameters = dict(self.model.named_parameters())
            deltas = [
                (named_parameters[name].detach() - initial).float().abs().mean()
                for name, initial in d2s_scale_start.items()
            ]
            result["d2s_scale_update_abs_mean"] = float(
                torch.stack(deltas).mean().item()
            )
        if s2d_scale_start:
            named_parameters = dict(self.model.named_parameters())
            deltas = [
                (named_parameters[name].detach() - initial).float().abs().mean()
                for name, initial in s2d_scale_start.items()
            ]
            result["s2d_scale_update_abs_mean"] = float(torch.stack(deltas).mean().item())
        named_parameters = dict(self.model.named_parameters())
        if shared_encoder_start:
            delta_squared = sum(
                float((named_parameters[name].detach().float() - initial.float()).square().sum().item())
                for name, initial in shared_encoder_start.items()
            )
            initial_squared = sum(
                float(initial.float().square().sum().item())
                for initial in shared_encoder_start.values()
            )
            result["encoder_parameter_relative_drift"] = math.sqrt(delta_squared) / max(math.sqrt(initial_squared), 1e-12)
        for direction, starts in (("d2s", d2s_mapping_start), ("s2d", s2d_mapping_start)):
            if starts:
                deltas = [
                    (named_parameters[name].detach() - initial).float().abs().mean()
                    for name, initial in starts.items()
                ]
                result[f"{direction}_mapping_update_abs_mean"] = float(torch.stack(deltas).mean().item())
        for level, interaction in sorted(self.model.interactions.items(), key=lambda item: int(item[0])):
            for name in ("seg_scale", "layer_scale", "vessel_scale"):
                value = getattr(interaction, name).detach().float()
                result[f"level{level}_{name}_signed_mean"] = float(value.mean().item())
                result[f"level{level}_{name}_abs_mean"] = float(value.abs().mean().item())
                result[f"level{level}_{name}_rms"] = float(value.square().mean().sqrt().item())
                result[f"level{level}_{name}_gradient_abs_mean"] = (
                    scale_gradient_totals[(level, name)] / max(optimizer_steps, 1)
                )
                result[f"level{level}_{name}_weight_decay"] = float(
                    self.config["train"].get("weight_decay", 0.0)
                )
                start = (s2d_scale_start if name == "seg_scale" else d2s_scale_start).get(
                    f"interactions.{level}.{name}"
                )
                if start is not None:
                    result[f"level{level}_{name}_update_abs_mean"] = float(
                        (value - start.float()).abs().mean().item()
                    )
            d2s_values = [
                parameter.detach().float().reshape(-1)
                for module in (
                    interaction.noise_head, interaction.restoration_context,
                    interaction.denoise_to_layer_gate, interaction.denoise_to_vessel_gate,
                    interaction.denoise_to_layer, interaction.denoise_to_vessel,
                ) for parameter in module.parameters()
            ]
            s2d_values = [
                parameter.detach().float().reshape(-1)
                for module in (
                    interaction.layer_anatomy, interaction.vessel_anatomy,
                    interaction.seg_to_denoise_gate,
                ) for parameter in module.parameters()
            ]
            if d2s_values:
                result[f"level{level}_d2s_mapping_parameter_rms"] = float(torch.cat(d2s_values).square().mean().sqrt().item())
                result[f"level{level}_d2s_mapping_gradient_norm"] = (
                    level_mapping_gradient_totals[(level, "d2s")] / max(optimizer_steps, 1)
                )
            if s2d_values:
                result[f"level{level}_s2d_mapping_parameter_rms"] = float(torch.cat(s2d_values).square().mean().sqrt().item())
                result[f"level{level}_s2d_mapping_gradient_norm"] = (
                    level_mapping_gradient_totals[(level, "s2d")] / max(optimizer_steps, 1)
                )
            for direction, starts in (("d2s", d2s_mapping_start), ("s2d", s2d_mapping_start)):
                level_starts = {
                    name: initial for name, initial in starts.items()
                    if name.startswith(f"interactions.{level}.")
                }
                if level_starts:
                    changes = [
                        (named_parameters[name].detach() - initial).float().abs().mean()
                        for name, initial in level_starts.items()
                    ]
                    result[f"level{level}_{direction}_mapping_update_abs_mean"] = float(
                        torch.stack(changes).mean().item()
                    )
        return result

    @torch.no_grad()
    def validate(
        self,
        loader: Optional[DataLoader] = None,
        description: str = "Validation",
        group_output: Optional[Path] = None,
    ) -> Dict[str, float]:
        loader = loader or self.val_loader
        evaluation_model = self.ema.module if self.ema is not None else self.model
        evaluation_model.eval()
        evaluation = self.config.get("evaluation", {})
        default_threshold = float(evaluation.get("threshold", 0.5))
        layer_threshold = float(
            evaluation.get("layer_threshold", default_threshold)
        )
        vessel_threshold = float(
            evaluation.get("vessel_threshold", default_threshold)
        )
        group_values = defaultdict(lambda: defaultdict(list))
        group_sample_ids = defaultdict(list)
        for batch in tqdm(loader, desc=description, leave=False):
            image = batch["image"].to(self.device, non_blocking=True)
            output = evaluation_model(
                image,
                return_features=False,
                return_auxiliary=False,
                interaction_guidance_image=batch.get("interaction_guidance", batch["image"]).to(self.device, non_blocking=True),
            )
            d2_teacher_layer_probability = None
            d2_teacher_vessel_probability = None
            if self.d2_teacher is not None:
                self.d2_teacher.eval()
                teacher_prediction = self.d2_teacher(
                    output["denoised"], return_features=False, return_auxiliary=False
                )
                d2_teacher_layer_probability = teacher_prediction["layer_prob"].cpu().numpy()
                d2_teacher_vessel_probability = teacher_prediction["vessel_prob"].cpu().numpy()
            d2s_disabled_vessel_probability = None
            if bool(self.config["train"].get("monitor_d2s_sensitivity", False)):
                interactions = list(evaluation_model.interactions.values())
                original_states = [
                    interaction.enable_denoise_to_seg
                    for interaction in interactions
                ]
                try:
                    for interaction in interactions:
                        interaction.enable_denoise_to_seg = False
                    disabled_output = evaluation_model(
                        image,
                        return_features=False,
                        return_auxiliary=False,
                        interaction_guidance_image=batch.get("interaction_guidance", batch["image"]).to(self.device, non_blocking=True),
                    )
                    d2s_disabled_vessel_probability = (
                        disabled_output["vessel_prob"].cpu().numpy()
                    )
                finally:
                    for interaction, enabled in zip(
                        interactions, original_states
                    ):
                        interaction.enable_denoise_to_seg = enabled
            layer_probability = output["layer_prob"].cpu().numpy()
            vessel_probability = output["vessel_prob"].cpu().numpy()
            layer = layer_probability >= layer_threshold
            vessel = vessel_probability >= vessel_threshold
            for index, group_id in enumerate(batch["group_id"]):
                d2_task_scores = []
                group_sample_ids[str(group_id)].append(
                    str(batch["sample_id"][index])
                )
                valid = batch["valid_mask"][index, 0].numpy() > 0.5
                if self.config.get("dose_response", {}).get("enabled", False):
                    valid &= batch["label_valid_mask"][index, 0].numpy() > 0.5
                if d2s_disabled_vessel_probability is not None:
                    group_values[group_id][
                        "d2s_vessel_probability_mean_abs_change"
                    ].append(
                        float(
                            np.abs(
                                vessel_probability[index, 0][valid]
                                - d2s_disabled_vessel_probability[index, 0][valid]
                            ).mean()
                        )
                    )
                if bool(batch["has_layer"][index]):
                    target = batch["layer_mask"][index, 0].numpy() > 0.5
                    layer_metrics = binary_metrics(
                        layer[index, 0][valid], target[valid]
                    )
                    for name in ("dice", "precision", "recall"):
                        group_values[group_id][f"layer_{name}"].append(
                            layer_metrics[name]
                        )
                    group_values[group_id]["layer_soft_dice"].append(
                        soft_dice_score(
                            layer_probability[index, 0], target, valid
                        )
                    )
                    if d2_teacher_layer_probability is not None:
                        d2_task_scores.append(soft_dice_score(
                            d2_teacher_layer_probability[index, 0], target, valid
                        ))
                if bool(batch["has_vessel"][index]):
                    target = batch["vessel_mask"][index, 0].numpy() > 0.5
                    vessel_valid = valid & (
                        batch["vessel_valid_mask"][index, 0].numpy() > 0.5
                    )
                    if d2_teacher_vessel_probability is not None:
                        d2_task_scores.append(soft_dice_score(
                            d2_teacher_vessel_probability[index, 0], target, vessel_valid
                        ))
                    if d2s_disabled_vessel_probability is not None:
                        group_values[group_id][
                            "d2s_disabled_vessel_soft_dice"
                        ].append(
                            soft_dice_score(
                                d2s_disabled_vessel_probability[index, 0],
                                target,
                                vessel_valid,
                            )
                        )
                    if bool(batch["has_layer"][index]):
                        diagnostics = vessel_diagnostic_metrics(
                            vessel_probability[index, 0],
                            layer_probability[index, 0],
                            target,
                            batch["layer_mask"][index, 0].numpy() > 0.5,
                            vessel_valid,
                            vessel_threshold=vessel_threshold,
                            layer_threshold=layer_threshold,
                            component_size_thresholds=(
                                tuple(evaluation["component_size_thresholds"])
                                if isinstance(
                                    evaluation.get("component_size_thresholds"),
                                    (list, tuple),
                                )
                                else None
                            ),
                            boundary_band_width=float(
                                evaluation.get("boundary_band_width", 3.0)
                            ),
                        )
                        for name, value in diagnostics.items():
                            group_values[group_id][name].append(value)
                    else:
                        vessel_metrics = binary_metrics(
                            vessel[index, 0][vessel_valid], target[vessel_valid]
                        )
                        for name, value in vessel_metrics.items():
                            group_values[group_id][f"vessel_{name}"].append(value)
                        group_values[group_id]["vessel_soft_dice"].append(
                            soft_dice_score(
                                vessel_probability[index, 0],
                                target,
                                vessel_valid,
                            )
                        )
                if d2_task_scores:
                    group_values[group_id]["teacher_task_preservation"].append(
                        float(np.mean(d2_task_scores))
                    )
                if bool(batch["has_clean"][index]) and not self.config.get("dose_response", {}).get("enabled", False):
                    prediction = output["denoised"][index, 0].cpu().numpy()
                    target = batch["clean"][index, 0].numpy()
                    mse = float(np.mean((prediction[valid] - target[valid]) ** 2))
                    group_values[group_id]["psnr"].append(
                        99.0 if mse < 1e-12 else 10.0 * math.log10(1.0 / mse)
                    )
        metrics = {}
        group_rows = []
        for group_id, values in sorted(group_values.items()):
            group_rows.append(
                {
                    "group_id": group_id,
                    "n_evaluated_frames": len(group_sample_ids[group_id]),
                    "sample_ids": ";".join(group_sample_ids[group_id]),
                    **{
                        name: float(np.mean(items))
                        for name, items in values.items()
                        if items
                    },
                }
            )
        names = sorted({name for values in group_values.values() for name in values})
        for name in names:
            per_group = [
                float(np.mean(values[name]))
                for values in group_values.values()
                if values[name]
            ]
            if per_group:
                metrics[name] = float(np.mean(per_group))
                metrics[f"n_groups_{name}"] = float(len(per_group))
        if group_output is not None:
            group_output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(group_rows).to_csv(
                group_output, index=False, encoding="utf-8-sig"
            )
        if self._denoise_probe_image is not None:
            current = evaluation_model(
                self._denoise_probe_image,
                return_features=False,
                return_auxiliary=False,
            )["denoised"]
            metrics["denoise_probe_max_abs_diff"] = float(
                (current - self._denoise_probe_reference).abs().max().item()
            )
            tolerance = self.config["train"].get("denoise_drift_tolerance")
            if tolerance is not None and metrics[
                "denoise_probe_max_abs_diff"
            ] > float(tolerance):
                raise RuntimeError(
                    "Frozen denoising function drifted: max_abs_diff="
                    f"{metrics['denoise_probe_max_abs_diff']:.8g} exceeds "
                    f"tolerance={float(tolerance):.8g}"
                )
        return metrics

    def fit(self) -> None:
        if self.config.get("dose_response", {}).get("enabled", False):
            from sabids.experiments.dose_response import dose_deterministic_algorithms
            with dose_deterministic_algorithms():
                self._fit_impl()
        else:
            self._fit_impl()

    def _fit_impl(self) -> None:
        epochs = int(self.config["train"].get("epochs", 100))
        patience = int(self.config["train"].get("early_stopping_patience", 30))
        monitor = self.config["train"].get("monitor", "vessel_dice")
        print(
            f"Device={self.device} | stage={self.stage} | "
            f"trainable_parameters={count_parameters(self.model):,} | "
            f"sampler={type(self.train_sampler).__name__} | "
            f"batch={self.config['train'].get('batch_size', 2)} x "
            f"accumulation={self.config['train'].get('gradient_accumulation_steps', 1)} | "
            f"memory_safe_joint={self.config['train'].get('memory_safe_joint', True)} | "
            f"stopgrad_repeat={self.config['train'].get('stopgrad_repeat_teacher', True)}"
        )
        diagnostics_dir = self.output_dir / "diagnostics"
        d2_selection_rows: list[dict] = []

        d2_global_optimizer_step = 0
        formal_teacher = bool(
            self.config.get("formal_d2_teacher", {}).get("enabled", False)
        )
        formal_teacher_global_optimizer_step = 0

        def checkpoint_extra(epoch_index: int) -> Dict:
            if self.phase_machine is None:
                extra = {"phase_state": None}
                if self.config.get("d2", {}).get("enabled", False):
                    extra["global_optimizer_step"] = int(d2_global_optimizer_step)
                elif formal_teacher:
                    extra["global_optimizer_step"] = int(
                        formal_teacher_global_optimizer_step
                    )
                return extra
            state = self.phase_machine.snapshot(epoch_index)
            state["batch_plan_state"].update({
                "train_loader_generator_state": (
                    self.train_loader.generator.get_state()
                    if self.train_loader.generator is not None else None
                ),
                "val_loader_generator_state": (
                    self.val_loader.generator.get_state()
                    if self.val_loader.generator is not None else None
                ),
            })
            state["active_parameter_groups"] = [
                name for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            ]
            extra = {"phase_state": state}
            if self.config.get("d2", {}).get("enabled", False):
                extra["global_optimizer_step"] = int(d2_global_optimizer_step)
            elif formal_teacher:
                extra["global_optimizer_step"] = int(
                    formal_teacher_global_optimizer_step
                )
            return extra

        if self.start_epoch == 0 and bool(
            self.config["train"].get("evaluate_epoch0", False)
        ):
            epoch0 = {
                "val": self.validate(
                    group_output=diagnostics_dir / "val_groups_epoch000.csv"
                )
            }
            if self.train_eval_loader is not None:
                epoch0["train_eval"] = self.validate(
                    loader=self.train_eval_loader,
                    description="Train-eval epoch 0",
                    group_output=diagnostics_dir / "train_groups_epoch000.csv",
                )
            write_json(epoch0, diagnostics_dir / "epoch000_metrics.json")
            save_checkpoint(
                self.output_dir / "epoch0.pth",
                self.model,
                self.optimizer,
                self.scheduler,
                -1,
                self.best_metric,
                self.config,
                self.scaler,
                self.ema.state_dict() if self.ema is not None else None,
                checkpoint_extra(-1),
            )
            print(f"Epoch 000 diagnostics: {epoch0}")
        completed_epochs = int(self.start_epoch)
        for epoch in range(self.start_epoch, epochs):
            start = time.time()
            train_metrics = self.train_epoch(epoch)
            if self.config.get("d2", {}).get("enabled", False):
                d2_global_optimizer_step += int(train_metrics.get("optimizer_steps", 0))
            elif formal_teacher:
                formal_teacher_global_optimizer_step += int(
                    train_metrics.get("optimizer_steps", 0)
                )
            epoch_number = epoch + 1
            val_metrics = self.validate(
                group_output=diagnostics_dir
                / f"val_groups_epoch{epoch_number:03d}.csv"
            )
            train_eval_metrics = {}
            if (
                self.train_eval_loader is not None
                and epoch_number % self.train_eval_every == 0
            ):
                train_eval_metrics = self.validate(
                    loader=self.train_eval_loader,
                    description=f"Train-eval {epoch_number}",
                    group_output=diagnostics_dir
                    / f"train_groups_epoch{epoch_number:03d}.csv",
                )
            monitored = val_metrics.get(monitor)
            if monitored is None:
                fallback = "layer_dice" if "layer_dice" in val_metrics else "psnr"
                monitored = val_metrics.get(fallback, -math.inf)
            if not math.isfinite(float(monitored)):
                raise FloatingPointError(
                    f"Validation monitor {monitor!r} is non-finite at epoch "
                    f"{epoch + 1}: {monitored!r}. Metrics={val_metrics}"
                )
            if self.scheduler_step_per_optimizer:
                pass
            elif self.scheduler_name == "plateau":
                self.scheduler.step(float(monitored))
            else:
                self.scheduler.step()
            improved = monitored > self.best_metric
            if improved:
                self.best_metric = monitored
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
            row = {
                "epoch": epoch + 1,
                "seconds": round(time.time() - start, 2),
                "lr": self.optimizer.param_groups[0]["lr"],
                "training_phase": (
                    "alternating"
                    if self.phase_machine is not None
                    and self.phase_machine.schedule == "order_alt"
                    and epoch < self.phase_machine.phase_epochs[0]
                    else self.phase_machine.phase(epoch)
                    if self.phase_machine is not None
                    else self.stage
                ),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{
                    f"train_eval_{k}": v
                    for k, v in train_eval_metrics.items()
                },
            }
            self.csv_logger.log(row)
            interaction_row = {
                key: value for key, value in row.items()
                if key in {"epoch", "training_phase"}
                or "interaction_" in key
                or "mapping_" in key
            }
            if len(interaction_row) > 2:
                CSVLogger(self.output_dir / "interaction_strength.csv").log(interaction_row)
            gradient_row = {
                key: value for key, value in row.items()
                if key in {"epoch", "training_phase"}
                or "gradient_" in key
                or "parameter_relative_drift" in key
            }
            if len(gradient_row) > 2:
                CSVLogger(self.output_dir / "gradient_audit.csv").log(gradient_row)
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, epoch + 1)
            ema_state = self.ema.state_dict() if self.ema is not None else None
            checkpoint_state = checkpoint_extra(epoch)
            save_checkpoint(
                self.output_dir / "last.pth",
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                self.best_metric,
                self.config,
                self.scaler,
                ema_state,
                checkpoint_state,
            )
            if self.config.get("d2", {}).get("enabled", False):
                if "psnr" not in val_metrics or "teacher_task_preservation" not in val_metrics:
                    if self.d2_teacher is not None:
                        raise RuntimeError("D2 validation lacks frozen-teacher preservation metric")
                d2_selection_rows.append({
                    "epoch": epoch_number,
                    "val_psnr": float(val_metrics.get("psnr", float("nan"))),
                    "val_teacher_task_preservation": float(
                        val_metrics.get("teacher_task_preservation", val_metrics.get("psnr", float("nan")))
                    ),
                })
                candidate_path = self.output_dir / "d2_selection_candidates" / f"epoch{epoch_number:03d}.pth"
                save_checkpoint(
                    candidate_path, self.model, self.optimizer, self.scheduler, epoch,
                    self.best_metric, self.config, self.scaler, ema_state, checkpoint_state,
                )
            if improved:
                best_path = self.output_dir / "best.pth"
                save_checkpoint(
                    best_path,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    self.best_metric,
                    self.config,
                    self.scaler,
                    ema_state,
                    checkpoint_state,
                )
                self._write_run_metadata(epoch + 1, monitor, best_path)
            print(
                f"Epoch {epoch + 1:03d} | monitor={monitored:.5f} | "
                f"best={self.best_metric:.5f} | bad_epochs={self.bad_epochs}"
            )
            completed_epochs = epoch + 1
            if self.bad_epochs >= patience:
                print("Early stopping triggered.")
                break
        if self.config.get("dose_response", {}).get("enabled", False):
            from sabids.experiments.dose_response import tensor_sha, write_strict_json
            dose_history = pd.read_csv(self.output_dir / "history.csv")
            initial = json.loads((self.output_dir / "initialization_audit.json").read_text(encoding="utf-8"))
            changed = {name: tensor_sha(parameter) != initial["tensor_sha256"][name]
                       for name, parameter in self.model.named_parameters()}
            changed_trainable = [n for n, p in self.model.named_parameters() if p.requires_grad and changed[n]]
            changed_frozen = [n for n, p in self.model.named_parameters() if not p.requires_grad and changed[n]]
            write_strict_json(self.output_dir / "dose_training_metadata.json", {
                "dose_response": self.config["dose_response"],
                "completed_epochs": epoch + 1,
                "completed_optimizer_steps": int(dose_history["train_optimizer_steps"].sum()),
                "expected_optimizer_steps": int(self.config["train"]["epochs"]) * math.ceil(
                    len(self.train_loader) / int(self.config["train"]["gradient_accumulation_steps"])),
                "training_validation_seconds": float(dose_history["seconds"].sum()),
                "changed_trainable_parameter_names": changed_trainable,
                "changed_frozen_parameter_names": changed_frozen,
                "primary_checkpoint": str(self.output_dir / "last.pth"),
                "primary_checkpoint_sha256": _sha256_file(self.output_dir / "last.pth"),
                "secondary_checkpoint": str(self.output_dir / "best.pth"),
                "coordinate_system": "model_grid_px", "original_resolution_boundary_metrics": "NOT IMPLEMENTED",
                "test_assets_opened": 0,
                "scientific_evaluation": self.config["dose_response"]["scientific_evaluation"],
                "notice": self.config["dose_response"].get("notice", "")})
            if changed_frozen or not changed_trainable:
                raise RuntimeError("Dose path update check failed; see dose_training_metadata.json")
        if self.config.get("d2", {}).get("enabled", False):
            if completed_epochs != epochs:
                raise RuntimeError("D2 fixed-budget run ended before checkpoint selection")
            teacher_audit_path = self.output_dir / "teacher_audit.json"
            teacher_audit = json.loads(teacher_audit_path.read_text(encoding="utf-8"))
            teacher_audit.update({
                "completed_epochs_audited": completed_epochs,
                "changed_parameter_count": 0,
                "requires_grad_parameter_count": int(
                    sum(parameter.numel() for parameter in self.d2_teacher.parameters() if parameter.requires_grad)
                    if self.d2_teacher is not None else 0
                ),
                "status": "passed" if self.d2_teacher is not None else "not_applicable",
            })
            write_json(teacher_audit, teacher_audit_path)
            # Fail closed before copying or binding any selected checkpoint.
            # A run with frozen drift or no trainable update must never leave a
            # provenance file that looks eligible for downstream use.
            from sabids.experiments.dose_response import tensor_sha, write_strict_json
            initialization = json.loads(
                (self.output_dir / "initialization_audit.json").read_text(encoding="utf-8")
            )
            changed_trainable, changed_frozen = [], []
            for name, parameter in self.model.named_parameters():
                changed = tensor_sha(parameter) != initialization["tensor_sha256"][name]
                if changed and parameter.requires_grad:
                    changed_trainable.append(name)
                elif changed:
                    changed_frozen.append(name)
            parameter_audit = {
                "status": "passed" if changed_trainable and not changed_frozen else "failed",
                "changed_trainable_parameter_names": changed_trainable,
                "changed_frozen_parameter_names": changed_frozen,
                "changed_trainable_parameter_count": len(changed_trainable),
                "changed_frozen_parameter_count": len(changed_frozen),
                "test_assets_opened": 0,
            }
            write_strict_json(self.output_dir / "parameter_audit.json", parameter_audit)
            if parameter_audit["status"] != "passed":
                raise RuntimeError("D2 parameter audit failed")
            import shutil
            from sabids.experiments.d2 import select_d2_checkpoints, write_d2_checkpoint_binding
            selection = select_d2_checkpoints(
                pd.DataFrame(d2_selection_rows),
                psnr_noninferiority_db=float(
                    self.config["d2"].get("selection", {}).get("psnr_noninferiority_db", 0.2)
                ),
            )
            selection["completed_epochs"] = completed_epochs
            d2_history = pd.read_csv(self.output_dir / "history.csv", low_memory=False)
            d2_selection_table = pd.DataFrame(d2_selection_rows)
            def d2_checkpoint_details(selected_epoch: int) -> Dict[str, float | int]:
                selected_rows = d2_history[pd.to_numeric(d2_history["epoch"]).eq(selected_epoch)]
                if len(selected_rows) != 1:
                    raise RuntimeError(f"D2 history does not uniquely contain epoch {selected_epoch}")
                metric_rows = d2_selection_table[
                    pd.to_numeric(d2_selection_table["epoch"]).eq(selected_epoch)
                ]
                if len(metric_rows) != 1:
                    raise RuntimeError(
                        f"D2 selection audit does not uniquely contain epoch {selected_epoch}"
                    )
                metric_row = metric_rows.iloc[0]
                through_epoch = d2_history[pd.to_numeric(d2_history["epoch"]).le(selected_epoch)]
                return {
                    "epoch": int(selected_epoch),
                    "global_optimizer_step": int(
                        pd.to_numeric(through_epoch["train_optimizer_steps"]).sum()
                    ),
                    "val_psnr": float(metric_row["val_psnr"]),
                    "val_teacher_task_preservation": float(
                        metric_row["val_teacher_task_preservation"]
                    ),
                }
            selection["checkpoint_details"] = {
                "best_pixel": d2_checkpoint_details(int(selection["best_pixel_epoch"])),
                "best_task_preserving": d2_checkpoint_details(
                    int(selection["best_task_preserving_epoch"])
                ),
                "last": d2_checkpoint_details(completed_epochs),
            }
            selection["fixed_before_training"] = self.config["d2"].get("selection", {})
            selection["test_assets_opened"] = 0
            selection_path = self.output_dir / "selection_history_audit.json"
            write_strict_json(selection_path, selection)
            selected = {
                "best_pixel": int(selection["best_pixel_epoch"]),
                "best_task_preserving": int(selection["best_task_preserving_epoch"]),
            }
            for kind, selected_epoch in selected.items():
                source = self.output_dir / "d2_selection_candidates" / f"epoch{selected_epoch:03d}.pth"
                destination = self.output_dir / f"{kind}.pth"
                if destination.exists():
                    raise FileExistsError(f"Refusing existing D2 selected checkpoint: {destination}")
                shutil.copy2(source, destination)
                write_d2_checkpoint_binding(
                    self._training_asset_initial_path,
                    destination,
                    self.output_dir / f"checkpoint_binding_{kind}.json",
                    kind,
                    selection,
                    selection_path,
                    self.output_dir / "teacher_audit.json",
                    self.output_dir / "parameter_audit.json",
                )
            history_table = pd.read_csv(self.output_dir / "history.csv", low_memory=False)
            write_strict_json(self.output_dir / "cost_profile.json", {
                "completed_epochs": completed_epochs,
                "training_validation_seconds": float(history_table["seconds"].sum()),
                "optimizer_steps": int(history_table["train_optimizer_steps"].sum()),
                "trainable_parameters": int(count_parameters(self.model)),
                "peak_cuda_memory_bytes": (
                    float(history_table["train_cuda_peak_memory_bytes"].max())
                    if "train_cuda_peak_memory_bytes" in history_table else None
                ),
                "checkpoint_candidate_count": len(d2_selection_rows),
                "memory_safe_d2_teacher": bool(
                    self.config.get("train", {}).get("memory_safe_d2_teacher", False)
                ),
                "d2_teacher_clean_precomputed_before_student": bool(
                    self.d2_teacher is not None
                ),
                "scientific_evaluation": bool(
                    self.config.get("d2", {}).get("scientific_evaluation", False)
                ),
                "notice": self.config.get("d2", {}).get(
                    "notice", "NOT FOR SCIENTIFIC EVALUATION"
                ),
                "test_assets_opened": 0,
            })
        if formal_teacher:
            if completed_epochs != epochs:
                raise RuntimeError("Formal D2 teacher ended before its fixed training budget")
            from sabids.experiments.dose_response import tensor_sha, write_strict_json
            initialization = json.loads(
                (self.output_dir / "initialization_audit.json").read_text(encoding="utf-8")
            )
            changed_trainable, changed_frozen = [], []
            trainable_names, frozen_names = [], []
            allowed_prefixes = (
                "adapters.layer.", "adapters.vessel.",
                "decoders.layer.", "decoders.vessel.",
                "layer_head.", "boundary_head.", "vessel_head.",
            )
            for name, parameter in self.model.named_parameters():
                changed = tensor_sha(parameter) != initialization["tensor_sha256"][name]
                if parameter.requires_grad:
                    trainable_names.append(name)
                    if changed:
                        changed_trainable.append(name)
                else:
                    frozen_names.append(name)
                    if changed:
                        changed_frozen.append(name)
            unexpected_trainable = [
                name for name in trainable_names
                if not name.startswith(allowed_prefixes)
            ]
            parameter_audit = {
                "status": "passed" if (
                    changed_trainable and not changed_frozen
                    and not unexpected_trainable
                    and formal_teacher_global_optimizer_step > 0
                ) else "failed",
                "trainable_parameter_names": trainable_names,
                "frozen_parameter_names": frozen_names,
                "unexpected_trainable_parameter_names": unexpected_trainable,
                "changed_trainable_parameter_names": changed_trainable,
                "changed_frozen_parameter_names": changed_frozen,
                "changed_trainable_parameter_count": len(changed_trainable),
                "changed_frozen_parameter_count": len(changed_frozen),
                "optimizer_steps": int(formal_teacher_global_optimizer_step),
                "completed_epochs": int(completed_epochs),
                "test_assets_opened": 0,
            }
            write_strict_json(
                self.output_dir / "formal_teacher_parameter_audit.json", parameter_audit
            )
            if parameter_audit["status"] != "passed":
                raise RuntimeError("Formal D2 teacher parameter audit failed")
            history_table = pd.read_csv(self.output_dir / "history.csv", low_memory=False)
            write_strict_json(self.output_dir / "formal_teacher_training_summary.json", {
                "status": "passed",
                "selection_rule": "best_validation_vessel_soft_dice",
                "completed_epochs": int(completed_epochs),
                "optimizer_steps": int(formal_teacher_global_optimizer_step),
                "best_checkpoint": str((self.output_dir / "best.pth").resolve()),
                "best_checkpoint_sha256": _sha256_file(self.output_dir / "best.pth"),
                "last_checkpoint": str((self.output_dir / "last.pth").resolve()),
                "last_checkpoint_sha256": _sha256_file(self.output_dir / "last.pth"),
                "history_rows": int(len(history_table)),
                "test_assets_opened": 0,
            })
        if self.config.get("training_asset_evidence", {}).get("enabled", False):
            from sabids.experiments.dose_response import (
                asset_inventory,
                bind_training_asset_evidence,
                resolve,
            )
            if self._training_asset_initial_path is None or self._training_asset_filtered is None:
                raise RuntimeError("Training asset evidence was not created before optimisation")
            evidence_cfg = self.config["training_asset_evidence"]
            project_root = Path(evidence_cfg.get("project_root", ".")).expanduser().resolve()
            data_root = resolve(project_root, self.config["data"].get("root") or project_root)
            current_records = asset_inventory(
                data_root,
                self._training_asset_filtered,
                include_labels=formal_teacher,
            )
            last_evidence = bind_training_asset_evidence(
                self._training_asset_initial_path,
                self.output_dir / "training_asset_inventory_last.json",
                self.output_dir / "last.pth",
                resolve(project_root, self.config["data"]["manifest"]),
                current_records,
                completed_epochs=completed_epochs,
                configured_epochs=epochs,
                selection_rule="fixed_final",
            )
            if self.config.get("d2", {}).get("enabled", False):
                from sabids.experiments.d2 import write_d2_checkpoint_binding
                selection = json.loads(
                    (self.output_dir / "selection_history_audit.json").read_text(encoding="utf-8")
                )
                write_d2_checkpoint_binding(
                    self._training_asset_initial_path,
                    self.output_dir / "last.pth",
                    self.output_dir / "checkpoint_binding_last.json",
                    "last", selection,
                    self.output_dir / "selection_history_audit.json",
                    self.output_dir / "teacher_audit.json",
                    self.output_dir / "parameter_audit.json",
                )
        self.writer.close()
