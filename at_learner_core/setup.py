#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages
from at_learner_core import __version__

VERSION = __version__

long_description = ""


def load_requirements(filename):
    with open(filename, "r") as f:
        return f.read().splitlines()


setup_info = dict(
    # Metadata
    name='at_learner_core',
    version=VERSION,
    author='Aleksandr Parkin',
    author_email='parkin.msu@gmail.com',
    url='https://github.com/AlexanderParkin/CASIA_CeFA_challenge/at_learner_core',
    description='A small package to learn models',
    long_description=long_description,
    license='BSD',

    # Package info
    packages=find_packages(),

    zip_safe=True,

    install_requires=load_requirements('requirements.txt')
)

setup(**setup_info)
