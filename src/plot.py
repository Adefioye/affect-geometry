from affect_geometry.cli_args import config_from_argv
from affect_geometry.pipeline import plot


def main() -> None:
    plot(config_from_argv("Plot affect-geometry results"))

if __name__ == "__main__":
    main()
