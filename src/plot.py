from affect_geometry.cli_args import run_command
from affect_geometry.pipeline import plot


def main() -> None:
    run_command("Plot affect-geometry results", plot)


if __name__ == "__main__":
    main()
