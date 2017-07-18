#!/usr/bin/env python3
import click

@click.group()
def main(args=None):
    pass

from .analyse import main as analyse
main.add_command(analyse)
from .categorize import main as categorize
main.add_command(categorize)
from .extract import main as extract
main.add_command(extract)
from .hmm import main as hmm
main.add_command(hmm)

if __name__ == "__main__":
    main()
