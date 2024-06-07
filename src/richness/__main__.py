"""Command-line parsing."""

import argparse
import sys
import time

from . import (
    abundance_richness_metrics,
    incidence_richness_metrics,
    read_frequencies,
)


def make_parser() -> argparse.ArgumentParser:
    """Creates ArgumentParser for command-line parsing."""

    parser = argparse.ArgumentParser(
        prog="richness",
        description=(
            "Nonparametric estimation of species richness"
            " from abundance or incidence data"
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    abundance_richness_parser = subparsers.add_parser(
        "abundance",
        help="Estimate species richness from abundance frequencies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    abundance_richness_parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.95,
        help="The confidence level of the confidence interval.",
    )
    abundance_richness_parser.add_argument(
        "-k",
        "--cutoff",
        type=int,
        default=10,
        help="Frequency cutoff for rare species used for estimating coverage.",
    )
    abundance_richness_parser.add_argument(
        "-d",
        "--disablecutoffadjust",
        action="store_true",
        help="Whether to disable cutoff adjustment in heterogeneous samples.",
    )
    abundance_richness_parser.add_argument(
        "frequencies", type=str, help="Path to an abundance frequency TSV."
    )

    incidence_richness_parser = subparsers.add_parser(
        "incidence",
        help="Estimate species richness from incidence frequencies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    incidence_richness_parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.95,
        help="The confidence level of the confidence interval.",
    )
    incidence_richness_parser.add_argument(
        "-n",
        type=int,
        default=1,
        help=(
            "If `n>1`, `raw_incidence` is interpreted as abundance frequencies"
            " which will be sampled randomly into `n` units."
        ),
    )
    incidence_richness_parser.add_argument(
        "-k",
        "--cutoff",
        type=int,
        default=10,
        help="Frequency cutoff for rare species used for estimating coverage.",
    )
    incidence_richness_parser.add_argument(
        "-d",
        "--disablecutoffadjust",
        action="store_true",
        help="Whether to disable cutoff adjustment in heterogeneous samples.",
    )
    incidence_richness_parser.add_argument(
        "raw_incidence",
        type=str,
        nargs="*",
        help="Path to frequencies or list of paths to raw incidence data.",
    )

    return parser


def main() -> None:
    """Parses arguments and prints richness estimates."""

    start = time.time()
    parser = make_parser()
    args = parser.parse_args()

    if args.command == "abundance":

        statistics, results = abundance_richness_metrics(
            read_frequencies(args.frequencies),
            confidence=args.confidence,
            cutoff=args.cutoff,
            adjust_cutoff=not args.disablecutoffadjust,
        )
        print(
            f"{statistics.to_string(float_format=lambda x: f'{x:.3f}')}"
            "\n\n"
            f"{results.to_string(float_format=lambda x: f'{x:.3f}')}"
            "\n\n"
            "See README.md for comparison on estimators."
            "\n"
            "tl;dr: If C>0.5 use Chao1 (or Chao1-bc if CV near zero);"
            " If CV>2 use ACE-1; ACE otherwise"
        )

    elif args.command == "incidence":

        statistics, results = incidence_richness_metrics(
            [read_frequencies(i) for i in args.raw_incidence],
            n=args.n,
            confidence=args.confidence,
            cutoff=args.cutoff,
            adjust_cutoff=not args.disablecutoffadjust,
        )
        print(
            f"{statistics.to_string(float_format=lambda x: f'{x:.3f}')}"
            "\n\n"
            f"{results.to_string(float_format=lambda x: f'{x:.3f}')}"
            "\n\n"
            "See README.md for comparison on estimators."
            "\n"
            "tl;dr: If C>0.5 use Chao2 (or Chao2-bc if CV near zero);"
            " If CV>2 use ICE-1; ICE otherwise"
        )

    else:
        parser.print_help(sys.stderr)
        sys.exit(1)

    print(f"\nElapsed time: {time.time() - start}")
    sys.exit(0)


if __name__ == "__main__":
    main()
