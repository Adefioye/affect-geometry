from affect_geometry.cli_args import run_command
from affect_geometry.pipeline import embed


def main() -> None:
    run_command("Embed representational dissimilarity matrices", embed)


if __name__ == "__main__":
    main()
