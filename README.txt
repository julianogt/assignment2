# Optimization Algorithms for Six Peaks and Knapsack Problem with Neural Network Classification

This project consists of two parts:
1. **Optimization Algorithms**: Solving random optimization problems (Six Peaks and Knapsack) using Randomized Hill Climbing (RHC), Simulated Annealing (SA), and Genetic Algorithm (GA).
2. **Shopping Behavior Prediction**: Using neural networks optimized with RHC, SA, and GA to classify customer shopping behavior based on frequency of purchases.

## Table of Contents

1. [Requirements](#requirements)
2. [Data Sources](#data-sources)
3. [Installation](#installation)
4. [Running the Code](#running-the-code)
5. [Results](#results)

## Requirements

- Python 3.7+
- mlrose-hiive (for optimization and neural network models)
- numpy
- pandas
- scikit-learn
- matplotlib

## Data Sources

1. **Shopping Behavior Dataset**: A dataset containing consumer behavior and shopping habits. [Download here](https://www.kaggle.com/datasets/zeesolver/consumer-behavior-and-shopping-habits-dataset).
2. **mlrose Library**: The mlrose library is used for optimization and neural networks with various algorithms. [Source here](https://github.com/gkhayes/mlrose).

## Installation

1. Clone this repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install dependencies:
    ```bash
    pip3 install -r requirements.txt
    ```

3. Place the `shopping_behavior.csv` file in the project directory.

## Running the Code

The project is split into two parts:

### Part 1: Optimization Algorithms for Six Peaks and Knapsack

The code for this part is in `part1.py`. It defines and solves the Six Peaks and Knapsack problems using three optimization algorithms: RHC, SA, and GA. To run the file:

```bash
python3 part1.py
```

### Part 2: Shopping Behavior Prediction with Neural Networks

The code for this part is in `part2.py`. It uses a neural network with RHC, SA, and GA optimization to classify customer behavior as high-frequency or low-frequency shoppers based on shopping data. To run the file:

```bash
python3 part2.py
```

