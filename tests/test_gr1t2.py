from __future__ import annotations

import json
from pathlib import Path

import newton
import numpy as np
import pytest
import warp as wp

from soma_retargeter.assets import bvh as bvh_utils
from soma_retargeter.assets.csv import (
    GR1T232DOF_CSVConfig,
    GR1T2_MUJOCO_JOINT_NAMES,
    get_csv_config,
    load_csv,
    save_csv,
)
from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline
from soma_retargeter.pipelines.utils import (
    SourceType,
    TargetType,
    build_and_validate_robot,
    get_retargeter_config,
    get_target_type_from_str,
    resolve_model_path,
)
from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer
from soma_retargeter.utils.space_conversion_utils import (
    SpaceConverter,
    get_facing_direction_type_from_str,
)


CIBO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE = (
    CIBO_ROOT
    / "data/bones_seed_sample/extracted/soma_uniform/bvh/221118"
    / "injured_R_leg_walk_ff_loop_360_004__A069.bvh"
)


def _gr1t2_config() -> dict:
    return get_retargeter_config(SourceType.SOMA, TargetType.GR1T2)


def test_target_registry_accepts_gr1t2() -> None:
    assert get_target_type_from_str("gr1t2") == TargetType.GR1T2
    assert get_csv_config("gr1t2").name == "gr1t2_32dof"


def test_unitree_g1_registry_and_model_behavior_remain_available() -> None:
    config = get_retargeter_config(SourceType.SOMA, TargetType.UNITREE_G1)
    builder = build_and_validate_robot(TargetType.UNITREE_G1, config)
    actuated = [
        joint_type for joint_type in builder.joint_type
        if joint_type == newton.JointType.REVOLUTE
    ]
    assert len(actuated) == 29
    assert get_csv_config("unitree_g1").name == "unitree_g1_29dof"


def test_model_path_expands_cibo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIBO_ROOT", str(CIBO_ROOT))
    path = resolve_model_path(TargetType.GR1T2, _gr1t2_config())
    assert path == (
        CIBO_ROOT
        / "cibo/contents/assets/robots/humanoid/gr1t2/mjcf/gr1t2_motion_32dof.xml"
    ).resolve()


def test_newton_model_has_verified_32_dof_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CIBO_ROOT", str(CIBO_ROOT))
    builder = build_and_validate_robot(TargetType.GR1T2, _gr1t2_config())
    names = [
        label.split("/")[-1].removeprefix("joint_") + "_joint"
        for label, joint_type in zip(builder.joint_label, builder.joint_type)
        if joint_type == newton.JointType.REVOLUTE
    ]
    assert names == list(GR1T2_MUJOCO_JOINT_NAMES)
    assert len(names) == len(set(names)) == 32


def test_ik_and_feet_bodies_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIBO_ROOT", str(CIBO_ROOT))
    config = _gr1t2_config()
    builder = build_and_validate_robot(TargetType.GR1T2, config)
    body_names = {label.split("/")[-1] for label in builder.body_label}
    feet_path = Path(__file__).parents[1] / "soma_retargeter/configs/gr1t2/gr1t2_feet_stabilizer_config.json"
    feet = json.loads(feet_path.read_text())
    mapped = {
        body
        for mapping in config["ik_map"].values()
        for body in (mapping["t_body"], mapping["r_body"])
    }
    assert mapped <= body_names
    assert set(feet["effectors"]) <= body_names


def test_gr1t2_csv_header_and_unit_round_trip(tmp_path: Path) -> None:
    config = GR1T232DOF_CSVConfig()
    assert config.csv_header[7:] == [
        f"{name}_dof" for name in GR1T2_MUJOCO_JOINT_NAMES
    ]
    raw = np.zeros((2, 39), dtype=np.float32)
    raw[:, :3] = [[1.25, -0.5, 0.95], [1.5, -0.25, 1.0]]
    raw[:, 3:7] = [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
    raw[:, 7:] = np.linspace(-0.25, 0.25, 64).reshape(2, 32)
    path = tmp_path / "motion.csv"
    save_csv(path, CSVAnimationBuffer.create_from_raw_data(raw, 120.0), config)
    loaded = load_csv(path, fps=120.0, csv_config=config)
    assert np.allclose(np.stack(loaded.data), raw, atol=1e-6)


def test_one_bvh_retargets_to_nonconstant_32_dof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not SAMPLE.is_file():
        pytest.skip("BONES-SEED sample is unavailable")
    monkeypatch.setenv("CIBO_ROOT", str(CIBO_ROOT))
    skeleton, animation = bvh_utils.load_bvh(SAMPLE)
    pipeline = NewtonPipeline(skeleton, "soma", "gr1t2")
    converter = SpaceConverter(get_facing_direction_type_from_str("Mujoco"))
    pipeline.add_input_motions(
        [animation], [converter.transform(wp.transform_identity())], True
    )
    output = pipeline.execute()
    assert output is not None and len(output) == 1
    data = np.stack(output[0].data)
    assert data.shape == (animation.num_frames, 39)
    assert np.isfinite(data).all()
    assert np.count_nonzero(np.ptp(data[:, 7:], axis=0) > 1e-6) >= 20
