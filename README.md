# ProtoTorch: Prototype Learning in PyTorch

![ProtoTorch Logo](https://prototorch.readthedocs.io/en/latest/_static/horizontal-lockup.png)

[![Build Status](https://travis-ci.org/si-cim/prototorch.svg?branch=master)](https://travis-ci.org/si-cim/prototorch)
![tests](https://github.com/si-cim/prototorch/workflows/tests/badge.svg)
[![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/si-cim/prototorch?color=yellow&label=version)](https://github.com/si-cim/prototorch/releases)
[![PyPI](https://img.shields.io/pypi/v/prototorch)](https://pypi.org/project/prototorch/)
[![codecov](https://codecov.io/gh/si-cim/prototorch/branch/master/graph/badge.svg)](https://codecov.io/gh/si-cim/prototorch)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/76273904bf9343f0a8b29cd8aca242e7)](https://www.codacy.com/gh/si-cim/prototorch?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=si-cim/prototorch&amp;utm_campaign=Badge_Grade)
![PyPI - Downloads](https://img.shields.io/pypi/dm/prototorch?color=blue)
[![GitHub license](https://img.shields.io/github/license/si-cim/prototorch)](https://github.com/si-cim/prototorch/blob/master/LICENSE)

*Tensorflow users, see:* [ProtoFlow](https://github.com/si-cim/protoflow)

## Description

This is a Python toolbox brewed at the Mittweida University of Applied Sciences
in Germany for bleeding-edge research in Prototype-based Machine Learning
methods and other interpretable models. The focus of ProtoTorch is ease-of-use,
extensibility and speed.

## Installation

ProtoTorch can be installed using `pip`.
```bash
pip install -U prototorch
```
To also install the extras, use
```bash
pip install -U prototorch[datasets,examples,tests]
```

To install the bleeding-edge features and improvements:
```bash
git clone https://github.com/si-cim/prototorch.git
git checkout dev
cd prototorch
pip install -e .
```

## Documentation

The documentation is available at <https://prototorch.readthedocs.io/en/latest/>

## Usage

### For researchers
ProtoTorch is modular. It is very easy to use the modular pieces provided by
ProtoTorch, like the layers, losses, callbacks and metrics to build your own
prototype-based(instance-based) models. These pieces blend-in seamlessly with
Keras allowing you to mix and match the modules from ProtoFlow with other
modules in `torch.nn`.

### For engineers
ProtoTorch comes prepackaged with many popular Learning Vector Quantization
(LVQ)-like algorithms in a convenient API. If you would simply like to be able
to use those algorithms to train large ML models on a GPU, ProtoTorch lets you
do this without requiring a black-belt in high-performance Tensor computing.


## Bibtex

If you would like to cite the package, please use this:
```bibtex
@misc{Ravichandran2020b,
  author = {Ravichandran, J},
  title = {ProtoTorch},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/si-cim/prototorch}}
}
