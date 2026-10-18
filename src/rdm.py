from affect_geometry.cli_args import run_command
from affect_geometry.pipeline import compute_rdms


def main() -> None:
    run_command("Compute representational dissimilarity matrices", compute_rdms)


if __name__ == "__main__":
    main()
