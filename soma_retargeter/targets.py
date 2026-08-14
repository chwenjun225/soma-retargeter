# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Central registry for retargeting targets and their local robot models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
import os
from pathlib import Path
from typing import Callable

import newton

import soma_retargeter.utils.io_utils as io_utils


class TargetType(IntEnum):
    """Supported robot targets."""

    UNITREE_G1 = auto()
    GR1T2 = auto()


def _unitree_g1_model() -> Path:
    return newton.utils.download_asset("unitree_g1") / "mjcf/g1_29dof_rev_1_0.xml"


_GR1T2_MODEL_RELATIVE = Path(
    "cibo/contents/assets/robots/humanoid/gr1t2/mjcf/gr1t2_motion_32dof.xml"
)


def _candidate_cibo_roots() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("CIBO_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(Path.cwd().resolve().parents)
    candidates.insert(0, Path.cwd().resolve())
    candidates.extend(Path(__file__).resolve().parents)
    return list(dict.fromkeys(candidates))


def _gr1t2_model() -> Path:
    for root in _candidate_cibo_roots():
        candidate = root / _GR1T2_MODEL_RELATIVE
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate the CIBO GR1T2 motion MJCF. Run from the CIBO checkout "
        "or set CIBO_ROOT to its absolute path."
    )


def _gr1t2_joint_order() -> tuple[str, ...]:
    try:
        from cibo.robots.gr1t2.joints import BODY_JOINTS_MUJOCO
    except ImportError as exc:
        raise RuntimeError(
            "GR1T2 requires the CIBO package so BODY_JOINTS_MUJOCO remains the "
            "single source of truth. Run from the CIBO checkout or install it editable."
        ) from exc
    return tuple(BODY_JOINTS_MUJOCO)


@dataclass(frozen=True)
class TargetDefinition:
    """All target-specific resources needed by the generic Newton pipeline."""

    target_type: TargetType
    name: str
    config_dir: str
    model_resolver: Callable[[], Path]
    retargeter_configs: dict[str, str]
    output_joint_order_resolver: Callable[[], tuple[str, ...]] | None = None

    def model_path(self) -> Path:
        """Resolve and validate the target MJCF path."""

        path = self.model_resolver().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Target MJCF does not exist: {path}")
        return path

    def retargeter_config_path(self, source: str) -> Path:
        """Return the source-to-target retargeter configuration path."""

        try:
            filename = self.retargeter_configs[source]
        except KeyError:
            allowed = ", ".join(sorted(self.retargeter_configs))
            raise ValueError(
                f"Unknown source type [{source}] for target [{self.name}]. "
                f"Allowed values: {allowed}"
            ) from None
        return io_utils.get_config_file(self.config_dir, filename)

    def output_joint_order(self) -> tuple[str, ...]:
        """Return the external CSV/SONIC joint order, if the target defines one."""

        if self.output_joint_order_resolver is None:
            return ()
        return self.output_joint_order_resolver()

    def build_model_builder(self) -> newton.ModelBuilder:
        """Load this target's MJCF into a fresh Newton builder."""

        builder = newton.ModelBuilder()
        builder.add_mjcf(self.model_path())
        self.validate_builder_joint_order(builder)
        return builder

    def validate_builder_joint_order(self, builder: newton.ModelBuilder) -> None:
        """Reject a target whose Newton coordinate order differs from its contract."""

        expected = self.output_joint_order()
        if not expected:
            return
        names = []
        for label, dof_dim in zip(builder.joint_label, builder.joint_dof_dim, strict=True):
            if dof_dim[1] != 1:
                continue
            raw = label.rsplit("/", 1)[-1].removeprefix("joint_")
            names.append(raw if raw.endswith("_joint") else f"{raw}_joint")
        if tuple(names) != expected:
            raise ValueError(
                f"Newton joint order for {self.name} does not match its output contract:\n"
                f"expected={list(expected)}\nactual={names}"
            )


_TARGETS = (
    TargetDefinition(
        target_type=TargetType.UNITREE_G1,
        name="unitree_g1",
        config_dir="unitree_g1",
        model_resolver=_unitree_g1_model,
        retargeter_configs={"soma": "soma_to_g1_retargeter_config.json"},
    ),
    TargetDefinition(
        target_type=TargetType.GR1T2,
        name="gr1t2",
        config_dir="gr1t2",
        model_resolver=_gr1t2_model,
        retargeter_configs={"soma": "soma_to_gr1t2_retargeter_config.json"},
        output_joint_order_resolver=_gr1t2_joint_order,
    ),
)

TARGETS_BY_TYPE = {target.target_type: target for target in _TARGETS}
TARGETS_BY_NAME = {target.name: target for target in _TARGETS}


def get_target(target: TargetType | str) -> TargetDefinition:
    """Look up one target by enum or stable CLI name."""

    registry = TARGETS_BY_NAME if isinstance(target, str) else TARGETS_BY_TYPE
    try:
        return registry[target]
    except KeyError:
        allowed = ", ".join(sorted(TARGETS_BY_NAME))
        raise ValueError(f"Unknown target type [{target}]. Allowed values: {allowed}") from None


def target_names() -> tuple[str, ...]:
    """Return all registered target names in deterministic order."""

    return tuple(target.name for target in _TARGETS)
