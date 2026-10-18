import json
from datetime import datetime, timedelta, timezone

import pytest

from affect_geometry.config import ExperimentConfig, ModelConfig, load_config, timestamp_run_id


def test_timestamp_run_id_is_utc_and_sortable() -> None:
    east_of_utc = timezone(timedelta(hours=2))
    instant = datetime(2026, 10, 18, 17, 30, 45, tzinfo=east_of_utc)

    assert timestamp_run_id(instant) == "20261018T153045Z"


def test_run_directory_is_namespaced_by_model_and_run() -> None:
    config = ExperimentConfig(
        run_id="20261018T153045Z",
        output_root="outputs",
        model=ModelConfig(name="Qwen/Qwen3-0.6B"),
    )

    assert config.run_dir.as_posix() == "outputs/qwen--qwen3-0.6b/20261018T153045Z"


def test_auto_run_id_requires_complete_pipeline_or_explicit_override(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"run_id": "auto"}))

    with pytest.raises(ValueError, match="Supply --run-id"):
        load_config(config_path)

    config = load_config(config_path, run_id="20261018T153045Z")
    assert config.run_id == "20261018T153045Z"
