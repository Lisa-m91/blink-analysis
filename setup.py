#!/usr/bin/env python3

from setuptools import setup

setup(name="Blink Analysis",
      version="0.0.2",
      description="A collection of scripts for analysing FRET/blinking.",
      packages=["blink_analysis"],
      requires=['tifffile (>=0.10.0)', 'numpy (>=1.10)', 'scipy (>= 0.18)'],
      entry_points={'console_scripts': ['blink_analysis=blink_analysis:entry']}
)
