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
    "torchvision>=0.7.1",
    "numpy>=1.9.1",
    "sklearn",
    "matplotlib",
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
    "torchinfo",
]
TESTS = ["codecov", "pytest"]
ALL = DATASETS + DEV + DOCS + EXAMPLES + TESTS

setup(
    name="prototorch",
    version="0.7.1",
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
    python_requires=">=3.6",
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(),
    zip_safe=False,
)
