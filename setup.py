"""

 ######
 #     # #####   ####  #####  ####  #####  ####  #####   ####  #    #
 #     # #    # #    #   #   #    #   #   #    # #    # #    # #    #
 ######  #    # #    #   #   #    #   #   #    # #    # #      ######
 #       #####  #    #   #   #    #   #   #    # #####  #      #    #
 #       #   #  #    #   #   #    #   #   #    # #   #  #    # #    #
 #       #    #  ####    #    ####    #    ####  #    #  ####  #    #

ProtoTorch Core Package
"""
from setuptools import find_packages, setup

PROJECT_URL = "https://github.com/si-cim/prototorch"
DOWNLOAD_URL = "https://github.com/si-cim/prototorch.git"

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

INSTALL_REQUIRES = [
    "torch>=2.0.0",
    "torchvision",
    "numpy",
    "scikit-learn",
    "matplotlib",
]
DATASETS = [
    "requests",
    "tqdm",
]
DEV = [
    "bump2version",
    "pre-commit",
]
DOCS = [
    "recommonmark",
    "sphinx",
    "sphinx_rtd_theme",
    "sphinxcontrib-katex",
    "sphinx-autodoc-typehints",
]
EXAMPLES = [
    "torchinfo",
]
TESTS = [
    "flake8",
    "pytest",
]
ALL = DATASETS + DEV + DOCS + EXAMPLES + TESTS

setup(
    name="prototorch",
    version="0.7.6",
    description="Highly extensible, GPU-supported "
    "Learning Vector Quantization (LVQ) toolbox "
    "built using PyTorch and its nn API.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jensun Ravichandran",
    author_email="jjensun@gmail.com",
    url=PROJECT_URL,
    download_url=DOWNLOAD_URL,
    license="MIT",
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "datasets": DATASETS,
        "dev": DEV,
        "docs": DOCS,
        "examples": EXAMPLES,
        "tests": TESTS,
        "all": ALL,
    },
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(),
    zip_safe=False,
)
