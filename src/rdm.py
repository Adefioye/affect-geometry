from affect_geometry.cli_args import config_from_argv
from affect_geometry.pipeline import compute_rdms


def main() -> None:
    compute_rdms(config_from_argv("Compute representational dissimilarity matrices"))

if __name__ == "__main__":
    main()
