import io
import os
import re

from pkg_resources import DistributionNotFound, get_distribution
from setuptools import find_packages, setup

# Package meta-data.
NAME = "qudida"
DESCRIPTION = "QUick and DIrty Domain Adaptation"
URL = "https://github.com/arsenyinfo/qudida"
REQUIRES_PYTHON = ">=3.5.0"

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
INSTALL_REQUIRES = ["numpy>=0.18.0", "scikit-learn>=0.19.1", "typing-extensions"]

# If none of packages in first installed, install second package
CHOOSE_INSTALL_REQUIRES = [
    (
        ("opencv-python>=4.0.1", "opencv-contrib-python>=4.0.1", "opencv-contrib-python-headless>=4.0.1"),
        "opencv-python-headless>=4.0.1",
    )
]


def choose_requirement(mains, secondary):
    """If some version version of main requirement installed, return main,
    else return secondary.

    Based ob https://github.com/albumentations-team/albumentations/blob/master/setup.py to be consistent with their
     dependency resolution approach.
    """
    chosen = secondary
    for main in mains:
        try:
            name = re.split(r"[!<>=]", main)[0]
            get_distribution(name)
            chosen = main
            break
        except DistributionNotFound:
            pass

    return str(chosen)


def get_install_requirements(install_requires, choose_install_requires):
    for mains, secondary in choose_install_requires:
        install_requires.append(choose_requirement(mains, secondary))
    return install_requires


def load_requirements(filename):
    try:
        # install from github
        with open(os.path.join(PROJECT_ROOT, filename), "r") as f:
            lineiter = f.read().splitlines()
        return [line for line in lineiter if line and not line.startswith("#")]
    except FileNotFoundError:
        # install from pypi with a minor chance requirements were not updated
        return ["opencv-python>=4.0.1", "numpy>=0.18.0", "scikit-learn>=0.19.1", "typing-extensions"]


def load_readme():
    readme_path = os.path.join(PROJECT_ROOT, "README.md")
    try:
        with io.open(readme_path, encoding="utf-8") as f:
            return "\n" + f.read()
    except FileNotFoundError:
        print('Readme not found :(')
        return ''


def load_version():
    context = {}
    with open(os.path.join(PROJECT_ROOT, "qudida", "__version__.py")) as f:
        exec(f.read(), context)
    return context["__version__"]


setup(
    name=NAME,
    version=load_version(),
    description=DESCRIPTION,
    long_description=load_readme(),
    long_description_content_type="text/markdown",
    keywords=[
        "Machine Learning",
        "Computer Vision",
    ],
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    install_requires=get_install_requirements(INSTALL_REQUIRES, CHOOSE_INSTALL_REQUIRES),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
