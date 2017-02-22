#!/usr/bin/env python3
from importlib import import_module

def main(args=None):
    from sys import argv
    if args is None:
        args = argv
    _, program, *args = args

    mod = import_module('.{}'.format(program), 'blink_analysis')
    mod.main(args)

if __name__ == "__main__":
    main()
