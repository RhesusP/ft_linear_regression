<div align="center">

# ft_linear_regression

#### An introduction to machine learning

</div>

## Requirements

- [numpy](https://numpy.org/install/)
- [matplotlib](https://matplotlib.org/stable/install/index.html)

## Installation

```bash
git clone https://github.com/RhesusP/ft_linear_regression.git
cd ft_linear_regression
```

## Usage

First, you need to train the model by running the following command:

```bash
python3 trainer.py
```

Then, you can use the model to make predictions by running the following command:

```bash
python3 predict.py
```

## About the project

This project is an introduction to machine learning. The goal is to predict the price of a car based on its mileage. The
model is trained using a linear regression algorithm.

General steps are :
1. Read the data from the `data.csv` file.
2. Perform a linear regression on the data.
3. Save $\theta_0$ and $\theta_1$ to a file.