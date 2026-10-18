from affect_geometry.cli_args import run_command
from affect_geometry.pipeline import analyze


def main() -> None:
    run_command("Analyze affect-geometry results", analyze)


if __name__ == "__main__":
    main()
