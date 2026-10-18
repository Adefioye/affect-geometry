from affect_geometry.cli_args import run_command
from affect_geometry.pipeline import prepare


def main() -> None:
    run_command("Prepare the experiment dataset", prepare)


if __name__ == "__main__":
    main()
