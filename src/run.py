from affect_geometry.cli_args import run_command
from affect_geometry.pipeline import run


def main() -> None:
    run_command("Run the complete affect-geometry pipeline", run, generate_run_id=True)


if __name__ == "__main__":
    main()
