from pathlib import Path

from affect_geometry.config import ExperimentConfig, ModelConfig
from affect_geometry.logging import configure_run_logging, get_logger, progress


def test_model_and_run_specific_log_contains_progress(tmp_path: Path) -> None:
    config = ExperimentConfig(
        run_id="20261018T153045Z",
        output_root=str(tmp_path / "outputs"),
        log_root=str(tmp_path / "logs"),
        model=ModelConfig(name="Qwen/Qwen3-0.6B"),
    )
    directory = configure_run_logging(config, "test")
    assert directory.name == "qwen--qwen3-0.6b__20261018T153045Z"
    assert list(progress(range(3), description="Testing progress", unit="step")) == [0, 1, 2]
    get_logger().info("test.message")
    for handler in get_logger().handlers:
        handler.flush()
    contents = (directory / "run.log").read_text()
    assert "model=Qwen/Qwen3-0.6B run_id=20261018T153045Z" in contents
    assert "progress.start description='Testing progress'" in contents
    assert "progress.complete description='Testing progress' completed=3 total=3" in contents
    assert "test.message" in contents
