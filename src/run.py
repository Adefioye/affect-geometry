from affect_geometry.cli_args import config_from_argv
from affect_geometry.pipeline import run


def main() -> None:
    run(config_from_argv("Run the complete affect-geometry pipeline", generate_run_id=True))

if __name__ == "__main__":
    main()
