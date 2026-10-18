from affect_geometry.cli_args import config_from_argv
from affect_geometry.pipeline import embed


def main() -> None:
    embed(config_from_argv("Embed representational dissimilarity matrices"))

if __name__ == "__main__":
    main()
