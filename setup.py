import sys

import setuptools


CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 6)

# This check and everything above must remain compatible with Python 2.7.
if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write('Python {} not supported. Minimum working tested version is {}'.format(CURRENT_PYTHON, REQUIRED_PYTHON))

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='elvis',
     version='0.1',
     author="Matteo A. Senese",
     description="A virtual laboratory for Language&Vision"
                 "developed by Matteo A. Senese",
     url="https://github.com/seo-95/ELVis",
     packages=['elvis'],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: Apache 2.0",
         "Operating System :: GNU-Linux",
     ],
 )
 
