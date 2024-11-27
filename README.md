<div align="center">

# ft_linear_regression

#### An introduction to machine learning

</div>

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [The theory](#the-theory)
  - [The dataset](#the-dataset)
  - [The model](#the-model)
  - [The cost function](#the-cost-function)
  - [Minimizing the cost function (Gradient Descent)](#minimizing-the-cost-function-gradient-descent)
    - [The Gradient Descent algorithm](#the-gradient-descent-algorithm)
    - [Learning rate](#learning-rate)
    - [Calculate a gradient](#calculate-a-gradient)
- [The practical part](#the-practical-part)
  - [The dataset](#the-dataset-1)
  - [The model (F)](#the-model-f)
  - [The cost function (J)](#the-cost-function-j)
  - [The gradient](#the-gradient)
  - [The Gradient Descent algorithm](#the-gradient-descent-algorithm-1)

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

## The theory

This project is an introduction to machine learning. The goal is to predict the price of a car based on its mileage. The
model is trained using a linear regression algorithm.

General steps are :

1. Read the data from the `data.csv` file.
2. Perform a linear regression on the data.
3. Save $\theta_0$ and $\theta_1$ to a file.
4. Use the model to make predictions.

### The dataset

The dataset is a CSV file containing two columns: `km` and `price`.

We represent the dataset as follows : $(x, y)$, with

- $x$ is the mileage of the car.
- $y$ is the price of the car.

From the dataset, we can deduce 2 important values:

- $m$ : the number of samples (rows) in the dataset (here, $m = 24$).
- $n$ : the number of features in the dataset (here, $n = 1$ = mileage).

### The model

The model is a linear regression model. It is represented by the following equation:

$$
f(x) = ax+ b
$$

The goal is to find the values of $a$ and $b$ that minimize the cost function $J(a, b)$.

### The cost function

The cost function measures errors between $a$ and $b$ and $y$ values in the dataset.
An error is calculated using the Euclidean distance between the predicted value and the actual value as follows:

$$
error = f(x^{(i)}) - y^{(i)}
$$

The sum of all errors is calculated as follows:

$$
J(a, b) = \frac{1}{2m} \sum_{i=1}^{m} (f(x^{(i)}) - y^{(i)})^2
$$

This is known as the [Mean Squared Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error).

### Minimizing the cost function (Gradient Descent)

The MSE is a sum of squared errors, so it have a convex shape with a single minimum. We are looking for this minimum with the [Gradient Descent algorithm](https://en.wikipedia.org/wiki/Gradient_descent).

#### The Gradient Descent algorithm

The Gradient Descent algorithm is as follows:

1. Initialize $a$ and $b$ to random values (0 in this project).
2. Calculate the derivative of this point.
3. Move with a step size $\alpha$ in the opposite direction of the derivative.
4. Repeat steps 2 and 3 until convergence.

#### Learning rate

The learning rate $\alpha$ is a hyperparameter that controls how much we are moving between iterations. This value is
arbitrary and can be changed to improve the model. However, if the learning rate is too high, the model may not converge
to the minimum. If the learning rate is too low, the model may take a long time to converge.

#### Calculate a gradient

For $a$ (or $\theta_1$) :

$$
\frac{\partial J}{\partial a} = \frac{1}{m} sum_{i=1}^{m} x (ax + b - y)
$$

For $b$ (or $\theta_0$) :

$$
\frac{\partial J}{\partial b} = \frac{1}{m} sum_{i=1}^{m} (ax + b - y)
$$

## The practical part

We are going to perform a linear regression on the dataset using matrix operations. This will allow us to calculate the
values of $a$ and $b$ in a single operation instead of iterating over the dataset.

### The dataset

We still have the same dataset $(x, y)$ as before.

### The model (F)

The model is represented by the following equation:

$$
F = X \theta
$$

Where:
$\theta$ is a vector containing the values of $a$ and $b$ : $$\begin{pmatrix}a\\\ b\end{pmatrix}$$
$X$ is a matrix containing the values of $x$ from the dataset and a column of 1s (for the bias) :

$$
\begin{pmatrix}x_1 & 1\\ x_2 & 1\\ \vdots & \vdots\\ x_m & 1\end{pmatrix}
$$

### The cost function (J)

We have

$$
J(a, b) = \frac{1}{2m} \sum_{i=1}^{m} (a x^{(i)} + b - y^{(i)})^2
$$

Where :

- $y^{(i)}$ is all the values of $y$ from the dataset.
- $a x^{(i)} + b$ is $X \theta$.
- the $\frac{1}{2m}$ and $^2$ are to simplify calculations.

So, the matrix form of the cost function is:

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (X \theta - Y)^2
$$

### The gradient

We have formulas for the two gradients:

$$
\frac{\partial J(a, b)}{\partial a} = \frac{1}{m} \sum_{i=1}^{m} x (a x + b - y)
$$

$$
\frac{\partial J(a, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (a x + b - y)
$$

We put these two gradients in a vector:

$$
\frac{\partial J(\theta)}{\partial \theta} = \begin{pmatrix}\frac{\partial J(a, b)}{\partial a}\\\ \frac{\partial J(a,
b)}{\partial b}\end{pmatrix}
$$

This vector calculate all the J derivatives for each $\theta$.  
We have can calculate the gradient with the following formula:  

$$
\frac{\partial J(\theta)}{\partial \theta} = \frac{1}{m} X^T (X \theta - Y)
$$

### The Gradient Descent algorithm

We can now apply the Gradient Descent algorithm to the matrix form of the cost function and loop until convergence.

We can calculate the new $\theta$ with the following formula:  

$$
\theta = \theta - \alpha \frac{\partial J}{\partial \theta}
$$