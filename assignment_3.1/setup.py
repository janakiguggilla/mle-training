# import os
import os.path as op
from distutils.core import setup

from setuptools import PEP420PackageFinder

ROOT = op.dirname(op.abspath(__file__))
SRC = op.join(ROOT, "src")

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    license = f.read()

setup(
    name="housing_price",
    version="0.0.1",
    package_dir={"": "src"},
    description="Housing price prediction",
    long_description=readme,
    author="JANAKI GUGGILLA",
    author_email='janaki.guggilla@tigeranalytics.com',
    license=license,
    packages=PEP420PackageFinder.find(where=str(SRC)),
)
