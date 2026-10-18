from affect_geometry.cli_args import run_command
from affect_geometry.pipeline import extract


def main() -> None:
    run_command("Extract model activations", extract)


if __name__ == "__main__":
    main()
