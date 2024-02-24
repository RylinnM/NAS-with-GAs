# Neural Architecture Search using Genetic Algorithm and Evolution Strategy

## Overview
This repository contains the implementation and evaluation of two algorithms: a Genetic Algorithm (GA) and an Evolution Strategy (ES), designed for Neural Architecture Search (NAS). The objective is to explore and identify high-performing neural architectures on the NAS-bench-101 benchmark. The project leverages IOHexperimenter for benchmark generation and IOHanalyzer for statistical analysis and visualization.

## Algorithms
1. **Genetic Algorithm (GA)**: Implemented in `Genetic Algorithm.py`, this script applies a genetic algorithm to the NAS problem. It includes functions for initialization, crossover, mutation, and various selection strategies (tournament, rank, and roulette wheel). 

2. **Evolution Strategy (ES)**: Located in `Evolution Strategy.py`, this script implements an evolution strategy approach for the NAS task. It provides mechanisms for parent selection, offspring generation, mutation, and survival selection.

## Requirements
- Python 3.x
- nasbench
- nas_ioh
- absl
- numpy

Ensure these libraries are installed. You can install them using pip:
```bash
pip install nasbench nas_ioh absl-py numpy
```

## Running the Code
To execute the algorithms, run the respective Python scripts from the command line:

For the Genetic Algorithm:

```bash
python Genetic Algorithm.py
```
For the Evolution Strategy:

```bash
python Evolution Strategy.py
```

Each script will perform multiple runs of the algorithm and output the best-found architecture and its performance.

## Evaluation
The performance of the algorithms is evaluated based on the average best-found fitness values and AUC values over 20 independent runs, each capped at 5,000 function evaluations.


