![Python workflow](https://github.com/spoopydale/cm_models/actions/workflows/python-package.yml/badge.svg) [![Documentation Status](https://readthedocs.org/projects/cm_models/badge/?version=latest)](https://cm-models.readthedocs.io/en/latest/?badge=latest)

# Models Package

A collection of models for electronic calculations.

Based on the PhD thesis of Christian Moulsdale: [Quantum properties of few-layer graphene and van der Waals heterostructures](https://research.manchester.ac.uk/en/studentTheses/quantum-properties-of-few-layer-graphene-and-van-der-waals-hetero).

Documentation is available at https://cm-models.readthedocs.io/en/latest.

## Installation

Install from github with Python>=3.10:

```bash
  pip install git+https://github.com/spoopydale/cm_models
```

## Features

Provides a simple, inheritance based system to create models for electronic calculations.

Specific features include:
 - Set model parameters
 - Build up the Hamiltonian from operators
 - Calculate electronic structure at zero and finite magnetic field
 - Include a moiré superlattice that reconstructs the band structure
 - Caches intermediate results between calculations to save on computation time

## Usage/Examples

The following script compares the Landau level fan diagrams of the 2- and 4-band models of bilayer graphene with a small bandgap (50meV) and requires matplotlib (``pip install matplotlib``):

```python
import numpy as np
from cm_models.blg import BLG, BLG2
import matplotlib.pyplot as plt

kwargs = dict(Delta_U=0.05)  # model parameters including a 50meV gap
K_plus = True  # valley
N = 50  # basis cutoff
B_min = 1.0  # minimum magnetic field [T] (>0 for stability)
B_max = 10.0  # maximum magnetic field [T]
P = 100  # number of magnetic field points
e_max = 0.1  # maximum energy

model = BLG(**kwargs)  # 4-band model
model2 = BLG2(**kwargs)  # 2-band model

B = np.linspace(B_min, B_max, P)  # magnetic field
e = model.e_levels(B, N=N, K_plus=K_plus)
e2 = model2.e_levels(B, N=N, K_plus=K_plus)

fig, axis = plt.subplots()
# plot one line separately for legend
axis.plot(B, e[:, 0], "C0-", label="4-band")
axis.plot(B, e2[:, 0], "C1:", label="2-band")
axis.plot(B, e[:, 1:], "C0-")
axis.plot(B, e2[:, 1:], "C1:")
axis.legend()
axis.set_xlabel(r"$B\,$[T]")
axis.set_ylabel(r"$\epsilon\,$[eV]")
axis.set_xlim(0.0, B_max)
axis.set_ylim(-e_max, e_max)
fig.tight_layout()

plt.show()
```
