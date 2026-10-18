from affect_geometry.cli_args import config_from_argv
from affect_geometry.pipeline import extract


def main() -> None:
    extract(config_from_argv("Extract model activations"))

if __name__ == "__main__":
    main()
