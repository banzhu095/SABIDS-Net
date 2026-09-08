from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch

@dataclass
class PhaseStateMachine:
    schedule: str
    phase_epochs: tuple[int, ...]
    optimizer_steps_per_epoch: int = 0
    global_step: int = 0
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def phase(self, epoch: int, batch_index: int = 0) -> str:
        if self.schedule == "order_alt" and epoch < self.phase_epochs[0]:
            # Alternate complete optimizer updates.  global_step changes only
            # after gradient accumulation has been stepped, so every microbatch
            # in one accumulation window uses the same task and parameter set.
            return "denoise" if self.global_step % 2 == 0 else "segment"
        if self.schedule == "order_alt":
            return "joint"
        first, second, _ = self.phase_epochs
        order = ("denoise", "segment") if self.schedule == "order_ds" else ("segment", "denoise")
        return order[0] if epoch < first else order[1] if epoch < first + second else "joint"

    def snapshot(self, epoch: int, batch_index: int = 0) -> dict[str, Any]:
        phase = self.phase(epoch, batch_index)
        if self.schedule == "order_alt":
            phase_epoch = epoch if phase != "joint" else epoch - self.phase_epochs[0]
        else:
            first, second, _ = self.phase_epochs
            phase_epoch = epoch if epoch < first else epoch - first if epoch < first + second else epoch - first - second
        if self.schedule == "order_alt":
            phase_step = self.global_step if phase != "joint" else self.global_step - self.phase_epochs[0] * self.optimizer_steps_per_epoch
        else:
            first, second, _ = self.phase_epochs
            offset_epochs = 0 if epoch < first else first if epoch < first + second else first + second
            phase_step = self.global_step - offset_epochs * self.optimizer_steps_per_epoch
        return {"schedule": self.schedule, "phase_epochs": list(self.phase_epochs),
                "current_phase": phase, "global_epoch": epoch,
                "global_step": self.global_step, "phase_step": max(0, phase_step),
                "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
                "phase_epoch": phase_epoch, "sampler_state": {"epoch": epoch},
                "batch_plan_state": {"next_global_step": self.global_step},
                "phase_transition_history": list(self.transitions),
                "rng_state_python": random.getstate(),
                "rng_state_numpy": np.random.get_state(), "rng_state_torch": torch.get_rng_state(),
                "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

    def restore(self, state: dict[str, Any]) -> None:
        if state.get("schedule") != self.schedule:
            raise RuntimeError(
                f"Cannot resume {self.schedule!r} from phase schedule "
                f"{state.get('schedule')!r}"
            )
        if tuple(state.get("phase_epochs", ())) != self.phase_epochs:
            raise RuntimeError("Cannot resume after changing schedule_epochs")
        saved_steps = int(state.get("optimizer_steps_per_epoch", self.optimizer_steps_per_epoch))
        if self.optimizer_steps_per_epoch and saved_steps != self.optimizer_steps_per_epoch:
            raise RuntimeError("Cannot resume after changing optimizer steps per epoch")
        self.global_step = int(state.get("global_step", 0))
        self.transitions = list(state.get("phase_transition_history", []))
        if "rng_state_python" in state:
            random.setstate(state["rng_state_python"])
        if "rng_state_numpy" in state:
            np.random.set_state(state["rng_state_numpy"])
        if "rng_state_torch" in state:
            torch.set_rng_state(state["rng_state_torch"])
        if torch.cuda.is_available() and state.get("rng_state_cuda"):
            torch.cuda.set_rng_state_all(state["rng_state_cuda"])

    def step(self) -> None:
        self.global_step += 1

    def record_epoch_phase(self, epoch: int) -> None:
        phase = (
            "alternating"
            if self.schedule == "order_alt" and epoch < self.phase_epochs[0]
            else self.phase(epoch)
        )
        if not self.transitions or self.transitions[-1]["phase"] != phase:
            self.transitions.append({"global_epoch": epoch, "global_step": self.global_step, "phase": phase})

    @staticmethod
    def set_trainable(model: torch.nn.Module, phase: str) -> list[str]:
        groups = {"shared": ("stem", "encoder_blocks", "downsamples"),
                  "denoise": ("adapters.denoise", "decoders.denoise", "residual_head"),
                  "segment": ("adapters.layer", "adapters.vessel", "decoders.layer", "decoders.vessel", "layer_head", "vessel_head", "boundary_head")}
        active = {"shared", "denoise"} if phase == "denoise" else {"shared", "segment"} if phase == "segment" else set(groups)
        names=[]
        for name,p in model.named_parameters():
            enabled=any(key in active and name.startswith(prefixes) for key,prefixes in groups.items())
            p.requires_grad_(enabled)
            if enabled: names.append(name)
        for key,prefixes in groups.items():
            for name,module in model.named_modules():
                if name.startswith(prefixes): module.train(key in active)
        return names
