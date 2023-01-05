
from setuptools import find_packages, setup

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='ML-FLOW',
    version='0.0.1',
    description='ML WORK FLOW',
    long_description=readme,
    author='JANAKI GUGGILLA',
    author_email='janaki.guggilla@tigeranalytics.com',
    url='https://github.com/janakiguggilla/mle-training',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)