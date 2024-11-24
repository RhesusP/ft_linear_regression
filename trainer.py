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


def descent_gradient(x_matrix, y_matrix, theta_matrix, learning_rate, pred_line, c_line):
    cost_history = np.empty((0, 1))
    iter = 0
    # Gradient descent while minimizing the cost function
    while True:
        print('iter: ', iter)
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
    print("Number of iterations: ", iter)
    return theta_matrix


# np.set_printoptions(suppress=True, precision=0)
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
y_copy = y

# Normalisation des données
x_min = np.min(x)
x_max = np.max(x)
x = (x - x_min) / (x_max - x_min)

# So we have m and n and we search for params a and b in the linear model f(x) = ax + b
# 1st step : F model
# X matrix
X = np.hstack((x, np.ones(x.shape)))  # Add a one column for bias

# theta
theta = np.ones((2, 1))

plt.ion()
figure, axis = plt.subplots(1, 2, figsize=(14, 7))

axis[0].scatter(x, y)
prediction_line = axis[0].plot(x, f_model(X, theta), c='r')
axis[0].set_title("Linear regression")
axis[0].set_xlabel("Mileage")
axis[0].set_ylabel("Price")

cost_line = axis[1].plot([], [], c='b')
axis[1].set_title("Cost evolution")
axis[1].set_xlabel("Number of iterations")
axis[1].set_ylabel("Cost (MSE)")

plt.tight_layout()
plt.show()

theta_final = descent_gradient(X, y, theta, 0.01, prediction_line[0], cost_line[0])

predictions = f_model(X, theta_final)
prediction_line[0].set_ydata(predictions)
plt.pause(0.005)

file = open(".theta", "w")
# Remove the standardization before saving thetas
theta_final[1] = theta_final[1] - theta_final[0] * x_min / (x_max - x_min)
theta_final[0] = theta_final[0] / (x_max - x_min)
file.write(str(theta_final[0][0]) + "\n" + str(theta_final[1][0]))

file.close()

plt.ioff()
plt.show()


# Coeficient de determination R^2
# https://youtu.be/vG6tDQc86Rs?t=1251


def r_squared(target, pred):
    u = ((target - pred) ** 2).sum()  # residu de la somme des carrés
    v = ((y - y.mean()) ** 2).sum()  # somme des carrés totale
    return 1 - u / v


print("R^2: ", r_squared(y, predictions))
