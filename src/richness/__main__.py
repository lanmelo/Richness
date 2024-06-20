"""Command-line parsing."""

import argparse
import sys
import time

from . import (
    abundance_richness_string,
    incidence_richness_string,
    read_frequencies,
)


def build_parser() -> argparse.ArgumentParser:
    """Creates ArgumentParser for command-line parsing."""
    parser = argparse.ArgumentParser(
        prog="richness",
        description=(
            "Nonparametric estimation of species richness"
            " from abundance or incidence data"
        ),
    )
    subparsers = parser.add_subparsers(dest="command")
    build_abundance_parser(subparsers)
    build_incidence_parser(subparsers)
    return parser


def build_abundance_parser(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    """Add the abundance subcommand to an ArgumentParser."""
    abundance_richness = subparsers.add_parser(
        "abundance",
        help="Estimate species richness from abundance frequencies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    abundance_richness.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.95,
        help="The confidence level of the confidence interval.",
    )
    abundance_richness.add_argument(
        "-k",
        "--cutoff",
        type=int,
        default=10,
        help="Frequency cutoff for rare species used for estimating coverage.",
    )
    abundance_richness.add_argument(
        "-d",
        "--disablecutoffadjust",
        action="store_true",
        help="Whether to disable cutoff adjustment in heterogeneous samples.",
    )
    abundance_richness.add_argument(
        "frequencies", type=str, help="Path to an abundance frequency TSV."
    )


def build_incidence_parser(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    """Add the incidence subcommand to an ArgumentParser."""
    incidence_parser = subparsers.add_parser(
        "incidence",
        help="Estimate species richness from incidence frequencies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    incidence_parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.95,
        help="The confidence level of the confidence interval.",
    )
    incidence_parser.add_argument(
        "-n",
        type=int,
        default=1,
        help=(
            "If `n>1`, `raw_incidence` is interpreted as abundance frequencies"
            " which will be sampled randomly into `n` units."
        ),
    )
    incidence_parser.add_argument(
        "-u",
        "--units",
        type=int,
        default=None,
        help=(
            "The number of sampling units. Used only if a single incidence"
            " frequency Series is provided and `n == 1`."
        ),
    )
    incidence_parser.add_argument(
        "-k",
        "--cutoff",
        type=int,
        default=10,
        help="Frequency cutoff for rare species used for estimating coverage.",
    )
    incidence_parser.add_argument(
        "-d",
        "--disablecutoffadjust",
        action="store_true",
        help="Whether to disable cutoff adjustment in heterogeneous samples.",
    )
    incidence_parser.add_argument(
        "raw_incidence",
        type=str,
        nargs="*",
        help="Path to frequencies or list of paths to raw incidence data.",
    )


def main() -> None:
    """Parses arguments and prints richness estimates."""

    start = time.time()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "abundance":
        print(
            abundance_richness_string(
                read_frequencies(args.frequencies),
                cutoff=args.cutoff,
                adjust_cutoff=not args.disablecutoffadjust,
                confidence=args.confidence,
            )
        )
    elif args.command == "incidence":
        print(
            incidence_richness_string(
                [read_frequencies(i) for i in args.raw_incidence],
                n=args.n,
                units=args.units,
                cutoff=args.cutoff,
                adjust_cutoff=not args.disablecutoffadjust,
                confidence=args.confidence,
            )
        )
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)

    print(f"\nElapsed time: {time.time() - start}")
    sys.exit(0)


if __name__ == "__main__":
    main()
