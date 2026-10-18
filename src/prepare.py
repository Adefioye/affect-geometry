from affect_geometry.cli_args import config_from_argv
from affect_geometry.pipeline import prepare


def main() -> None:
    prepare(config_from_argv("Prepare the experiment dataset"))

if __name__ == "__main__":
    main()
