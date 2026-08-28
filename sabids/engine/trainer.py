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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_label_asset(path: Path) -> Dict[str, object]:
    raw_sha256 = _sha256_file(path)
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


def build_loaders(config: Dict) -> tuple[DataLoader, DataLoader, object]:
    data_cfg = config["data"]
    train_dataset = OCTManifestDataset(
        data_cfg["manifest"],
        split=data_cfg.get("train_split", "train"),
        transform=_make_transform(config, True),
        sample_repeat=True,
        root=data_cfg.get("root"),
        datasets=data_cfg.get("train_datasets"),
        groups=data_cfg.get("train_groups"),
    )
    val_dataset = OCTManifestDataset(
        data_cfg["manifest"],
        split=data_cfg.get("val_split", "val"),
        transform=_make_transform(config, False),
        sample_repeat=False,
        root=data_cfg.get("root"),
        datasets=data_cfg.get("val_datasets"),
        groups=data_cfg.get("val_groups"),
    )
    max_val_samples = data_cfg.get("max_val_samples")
    if max_val_samples is not None:
        count = min(int(max_val_samples), len(val_dataset))
        val_dataset = Subset(val_dataset, range(count))
    vessel_fraction = float(data_cfg.get("vessel_oversample_fraction", 0.0) or 0.0)
    if vessel_fraction > 0.0:
        train_sampler = SparseAnnotationSampler(
            train_dataset,
            vessel_fraction=vessel_fraction,
            samples_per_epoch=data_cfg.get("samples_per_epoch"),
            seed=int(config.get("seed", 42)),
        )
    else:
        train_sampler = GroupUniformSampler(
            train_dataset,
            samples_per_epoch=data_cfg.get("samples_per_epoch"),
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
    )


class Trainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = get_device(config.get("device", "auto"))
        seed_everything(
            int(config.get("seed", 42)),
            bool(config.get("deterministic", False)),
            use_cuda=self.device.type == "cuda",
        )
        self.output_dir = Path(config["train"].get("output_dir", "runs/sabids"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_loader, self.val_loader, self.train_sampler = build_loaders(config)
        self._record_run_inputs()
        if str(config.get("train", {}).get("stage", "")) == "interaction":
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
        scheduler_name = str(config["train"].get("scheduler", "cosine"))
        self.scheduler_name = scheduler_name
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
                T_max=max(epochs, 1),
                eta_min=float(config["train"].get("minimum_learning_rate", 1e-6)),
            )
        else:
            raise ValueError("train.scheduler must be cosine or plateau")
        amp_enabled = bool(config["train"].get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
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
                "data_plan_sha256": hashlib.sha256(plan_payload).hexdigest(),
                "data_rng_seed": int(self.config.get("seed", 42)) + 1_000_003,
                "model_rng_seed": int(self.config.get("seed", 42)),
            },
            self.output_dir / "initialization_audit.json",
        )

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
        asset_table = table
        if str(self.config.get("train", {}).get("stage", "")) == "interaction":
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
                inspection = _inspect_label_asset(asset) if asset.is_file() else {}
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
        if resume:
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
            if self.ema is not None:
                if checkpoint.get("ema") is not None:
                    self.ema.load_state_dict(checkpoint["ema"])
                else:
                    # Older checkpoints may not contain an EMA state. Start the
                    # teacher from the restored student instead of a random copy.
                    self.ema.load_state_dict(self.model.state_dict())
        elif pretrained:
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
                raw_clean = self.model(
                    identity_input,
                    detach_cross=detach_cross,
                    return_features=False,
                    return_auxiliary=False,
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
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.train_sampler.set_epoch(epoch)
        totals = defaultdict(float)
        steps = 0
        optimizer_steps = 0
        gradient_norm_total = 0.0
        seen_groups = set()
        layer_supervised_samples = 0
        vessel_supervised_samples = 0
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
        accumulation_steps = max(
            1, int(self.config["train"].get("gradient_accumulation_steps", 1))
        )
        detach_epochs = int(self.config["train"].get("detach_cross_epochs", 10))
        detach_cross = epoch < detach_epochs
        ramp_epochs = max(int(self.config["train"].get("ramp_epochs", 20)), 1)
        ramp = min(1.0, max(0.0, (epoch + 1) / ramp_epochs))
        progress = tqdm(self.train_loader, desc=f"Train {epoch + 1}", leave=False)
        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(progress):
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
                repeat_output, clean_output, teacher_output = self._forward_auxiliary(
                    batch, detach_cross
                )
                output = self.model(batch["image"], detach_cross=detach_cross)
                losses = self.loss_fn(
                    output,
                    batch,
                    stage=self.stage,
                    repeat_output=repeat_output,
                    clean_output=clean_output,
                    teacher_output=teacher_output,
                    ramp=ramp,
                )
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
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_steps += 1
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
                    "guidance_finite",
                    "denoise_guidance_mean",
                    "denoise_guidance_std",
                    "denoise_guidance_finite",
                ):
                    values = [float(item[name].item()) for item in auxiliary if name in item]
                    if values:
                        totals[f"interaction_{name}"] += float(np.mean(values))
                    for item in auxiliary:
                        if name in item and "level" in item:
                            level = int(item["level"].item())
                            totals[
                                f"interaction_level{level}_{name}"
                            ] += float(item[name].item())
            progress.set_postfix(loss=totals["total"] / steps)
            del output, repeat_output, clean_output, teacher_output, losses
        result = {key: value / max(steps, 1) for key, value in totals.items()}
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
            }
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
        for direction, starts in (("d2s", d2s_mapping_start), ("s2d", s2d_mapping_start)):
            if starts:
                deltas = [
                    (named_parameters[name].detach() - initial).float().abs().mean()
                    for name, initial in starts.items()
                ]
                result[f"{direction}_mapping_update_abs_mean"] = float(torch.stack(deltas).mean().item())
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
                image, return_features=False, return_auxiliary=False
            )
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
                        image, return_features=False, return_auxiliary=False
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
                group_sample_ids[str(group_id)].append(
                    str(batch["sample_id"][index])
                )
                valid = batch["valid_mask"][index, 0].numpy() > 0.5
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
                if bool(batch["has_vessel"][index]):
                    target = batch["vessel_mask"][index, 0].numpy() > 0.5
                    vessel_valid = valid & (
                        batch["vessel_valid_mask"][index, 0].numpy() > 0.5
                    )
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
                if bool(batch["has_clean"][index]):
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
            )
            print(f"Epoch 000 diagnostics: {epoch0}")
        for epoch in range(self.start_epoch, epochs):
            start = time.time()
            train_metrics = self.train_epoch(epoch)
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
            if self.scheduler_name == "plateau":
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
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{
                    f"train_eval_{k}": v
                    for k, v in train_eval_metrics.items()
                },
            }
            self.csv_logger.log(row)
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, epoch + 1)
            ema_state = self.ema.state_dict() if self.ema is not None else None
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
                )
                self._write_run_metadata(epoch + 1, monitor, best_path)
            print(
                f"Epoch {epoch + 1:03d} | monitor={monitored:.5f} | "
                f"best={self.best_metric:.5f} | bad_epochs={self.bad_epochs}"
            )
            if self.bad_epochs >= patience:
                print("Early stopping triggered.")
                break
        self.writer.close()
