import csv
import numpy as np
import matplotlib.pyplot as plt


# F = X.theta
def f_model(x_matrix, theta_matrix):
    return x_matrix.dot(theta_matrix)


# Fonction cout J : error measurement (MSE)
def cost(x_matrix, y_matrix, theta_matrix):
    m = len(y_matrix)
    return 1 / (2 * m) * np.sum((f_model(x_matrix, theta_matrix) - y_matrix) ** 2)


# Algo de minimisation : we want to minimize errors thanks to the gradient descent algorithm
# algo de la descente de gradient
def gradient(x_matrix, y_matrix, theta_matrix):
    m = len(y_matrix)
    return 1 / m * x_matrix.T.dot(f_model(x_matrix, theta_matrix) - y_matrix)


def descent_gradient(x_matrix, y_matrix, theta_matrix, learning_rate, nb_iter):
    history = np.zeros(nb_iter)
    for i in range(0, nb_iter):
        theta_matrix = theta_matrix - learning_rate * gradient(x_matrix, y_matrix, theta_matrix)
        history[i] = cost(x_matrix, y_matrix, theta_matrix)
    return theta_matrix, history


np.set_printoptions(suppress=True, precision=0)
file_path = "data.csv"

x = np.empty((0, 1))
y = np.empty((0, 1))

# Read the dataset
with open(file_path, "r") as file:
    csvreader = csv.reader(file)
    i = 0
    for idx_row, row in enumerate(csvreader):
        if idx_row > 0:
            x = np.vstack([x, [int(row[0])]])
            y = np.vstack([y, [int(row[1])]])

x_copy = x
print("x: ", x)

# Normalisation des données
mean_x = np.mean(x)
std_x = np.std(x)
x = (x - mean_x) / std_x
print("normalized x: ", x)

# So we have m and n and we search for params a and b in the linear model f(x) = ax + b
# 1st step : F model
# X matrix
X = np.hstack((x, np.ones(x.shape)))  # Add a one column for bias

# theta
theta = np.ones((2, 1))
theta_final, cost_history = descent_gradient(X, y, theta, learning_rate=0.1, nb_iter=50)
print("theta final: ", theta_final)

predictions = f_model(X, theta_final)
figure, axis = plt.subplots(1, 2, figsize=(14, 7))

axis[0].scatter(x_copy, y)
axis[0].plot(x_copy, predictions, c='r')
axis[0].set_title("Linear regression")
axis[0].set_xlabel("Mileage")
axis[0].set_ylabel("Price")

axis[1].plot(range(len(cost_history)), cost_history)
axis[1].set_title("Cost evolution")
axis[1].set_xlabel("Number of iterations")
axis[1].set_ylabel("Cost (MSE)")

plt.tight_layout()
plt.show()
