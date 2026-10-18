from affect_geometry.cli_args import config_from_argv
from affect_geometry.pipeline import analyze


def main() -> None:
    analyze(config_from_argv("Analyze affect-geometry results"))

if __name__ == "__main__":
    main()
