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

with open("README.md", "r") as fh:
    long_description = fh.read()

INSTALL_REQUIRES = [
    "torch>=1.3.1",
    "torchvision>=0.6.0",
    "numpy>=1.9.1",
    "sklearn",
]
DATASETS = [
    "requests",
    "tqdm",
]
DEV = [
    "bumpversion",
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
    "matplotlib",
    "torchinfo",
]
TESTS = ["codecov", "pytest"]
ALL = DATASETS + DEV + DOCS + EXAMPLES + TESTS

setup(
    name="prototorch",
    version="0.6.0",
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
    python_requires=">=3.7",
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
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
    zip_safe=False,
)
