import csv
import numpy as np
import matplotlib.pyplot as plt


"""
Calculate the model F = X.theta
Parameters:
    - x_matrix: matrix of features (input data)
    - theta_matrix: matrix of parameters (weights of the model)
"""
def f_model(x_matrix, theta_matrix):
    return x_matrix.dot(theta_matrix)


"""
Calculate the cost function J(theta) = 1/2m * sum((X * theta) - Y)^2
This function is used to measure the error of the model thanks to the MSE (Mean Squared Error)
Parameters:
    - x_matrix: matrix of features
    - y_matrix: matrix of target values
    - theta_matrix: matrix of parameters
"""
def cost(x_matrix, y_matrix, theta_matrix):
    m = len(y_matrix)
    return 1 / (2 * m) * np.sum((f_model(x_matrix, theta_matrix) - y_matrix) ** 2)


"""
Calculate a gradient
Parameters:
    - x_matrix: matrix of features
    - y_matrix: matrix of target values
    - theta_matrix: matrix of parameters
"""
def gradient(x_matrix, y_matrix, theta_matrix):
    m = len(y_matrix)
    return 1 / m * x_matrix.T.dot(f_model(x_matrix, theta_matrix) - y_matrix)


"""
Gradient descent algorithm
This is a minimization algorithm that aims to find the best parameters theta that minimize the cost function.
The algorithm loops until the cost difference between two iterations is negligible.
Parameters:
    - x_matrix: matrix of features
    - y_matrix: matrix of target values
    - theta_matrix: matrix of parameters
    - learning_rate: the step between each iteration
    - pred_line: line on the chart representing the model prediction (F)
    - c_line: line ont the chart representing the cost evolution
Returns:
    - the best theta that minimize the cost function
"""
def gradient_descent(x_matrix, y_matrix, theta_matrix, learning_rate, pred_line, c_line):
    cost_history = np.empty((0, 1))
    iter = 0
    while True:
        theta_matrix = theta_matrix - learning_rate * gradient(x_matrix, y_matrix, theta_matrix)
        if iter % 100 == 0:
            pred_line.set_ydata(f_model(x_matrix, theta_matrix))
            cost_value = cost(x_matrix, y_matrix, theta_matrix)
            cost_history = np.vstack([cost_history, [cost_value]])
            c_line.set_xdata(np.arange(len(cost_history)) * 100)
            c_line.set_ydata(cost_history)
            axis[1].relim()
            axis[1].autoscale_view()
            plt.pause(0.005)
        # Stop the loop if the cost function converges
        if len(cost_history) > 1 and cost_history[-2] - cost_history[-1] < 0.001:
            break
        iter += 1
    print("🔄 Number of iterations: ", iter)
    return theta_matrix


"""
Calculate the performance of the model with the R^2 coefficient
Parameters:
    - target: matrix of target values
    - pred: matrix of predicted values
"""
def r_squared(target, pred):
    u = ((target - pred) ** 2).sum()  # residu de la somme des carrés
    v = ((y - y.mean()) ** 2).sum()  # somme des carrés totale
    return 1 - u / v


# ------------------- Main -------------------
file_path = "data.csv"

x = np.empty((0, 1))
y = np.empty((0, 1))

# Read the dataset
try:
    with open(file_path, "r") as file:
        csvreader = csv.reader(file)
        i = 0
        for idx_row, row in enumerate(csvreader):
            if idx_row > 0:
                x = np.vstack([x, [int(row[0])]])
                y = np.vstack([y, [int(row[1])]])
except Exception as e:
    print("Error: Unable to read the dataset (", e, ")")
    exit(1)

x_copy = x
y_copy = y

# Standardization
x_min = np.min(x)
x_max = np.max(x)
x = (x - x_min) / (x_max - x_min)

# X matrix = x with bias
X = np.hstack((x, np.ones(x.shape)))  # Add a one column for bias

# Theta matrix = [theta0, theta1] = [0, 0]
theta = np.ones((2, 1))

plt.ion()
figure, axis = plt.subplots(1, 2, figsize=(14, 7))
# First plot (dataset + model)
axis[0].scatter(x, y)
prediction_line = axis[0].plot(x, f_model(X, theta), c='r')
axis[0].set_title("Linear regression")
axis[0].set_xlabel("Mileage")
axis[0].set_ylabel("Price")
# Second plot (cost evolution)
cost_line = axis[1].plot([], [], c='b')
axis[1].set_title("Cost evolution")
axis[1].set_xlabel("Number of iterations")
axis[1].set_ylabel("Cost (MSE)")
plt.tight_layout()
plt.show()

theta_final = gradient_descent(X, y, theta, 0.01, prediction_line[0], cost_line[0])

predictions = f_model(X, theta_final)
prediction_line[0].set_ydata(predictions)
plt.pause(0.005)

try:
    file = open(".theta", "w")
    # Remove the standardization before saving thetas
    theta_final[1] = theta_final[1] - theta_final[0] * x_min / (x_max - x_min)
    theta_final[0] = theta_final[0] / (x_max - x_min)
    file.write(str(theta_final[0][0]) + "\n" + str(theta_final[1][0]))
    file.close()
except Exception as e:
    print("Error: Unable to save thetas (", e, ")")

plt.ioff()
plt.show()

print("⚡ Performance: ", r_squared(y, predictions) * 100, "%")
