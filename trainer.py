import csv
import numpy as np
import matplotlib.pyplot as plt


def f_model(x_matrix: np.ndarray, theta_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the model F = X.theta
    :param x_matrix: matrix of features (input data)
    :param theta_matrix: matrix of parameters (weights of the model)
    :return:
    """
    return x_matrix.dot(theta_matrix)


def cost(x_matrix: np.ndarray, y_matrix: np.ndarray, theta_matrix: np.ndarray) -> float:
    """
    Calculate the cost function J(theta) = 1/2m * sum((X * theta) - Y)^2
    This function is used to measure the error of the model thanks to the MSE (Mean Squared Error)
    :param x_matrix: matrix of features
    :param y_matrix: matrix of target values
    :param theta_matrix: matrix of parameters
    :return:
    """
    m = len(y_matrix)
    return 1 / (2 * m) * np.sum((f_model(x_matrix, theta_matrix) - y_matrix) ** 2)


def gradient(x_matrix: np.ndarray, y_matrix: np.ndarray, theta_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate a gradient
    :param x_matrix: matrix of features
    :param y_matrix: matrix of target values
    :param theta_matrix: matrix of parameters
    :return:
    """
    m = len(y_matrix)
    return 1 / m * x_matrix.T.dot(f_model(x_matrix, theta_matrix) - y_matrix)


def gradient_descent(x_matrix: np.ndarray, y_matrix: np.ndarray, theta_matrix: np.ndarray, learning_rate: float,
                     axis: plt.Axes, pred_line: plt.Line2D, c_line: plt.Line2D) -> np.ndarray:
    """
    Gradient descent algorithm
    This is a minimization algorithm that aims to find the best parameters theta that minimize the cost function.
    The algorithm loops until the cost difference between two iterations is negligible.
    :param x_matrix: matrix of features
    :param y_matrix: matrix of target values
    :param theta_matrix: matrix of parameters
    :param learning_rate: the step between each iteration
    :param axis: the chart axis
    :param pred_line: line on the chart representing the model prediction (F)
    :param c_line: line on the chart representing the cost evolution
    :return: the best theta that minimize the cost function
    """
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
        if len(cost_history) > 1 and cost_history[-2] - cost_history[-1] < 0.01:
            break
        iter += 1
    print("🔄 Number of iterations: ", iter)
    return theta_matrix


def r_squared(target: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate the performance of the model with the R^2 coefficient
    :param target: matrix of target values
    :param pred: matrix of predicted values
    :return:
    """
    u = ((target - pred) ** 2).sum()  # residu de la somme des carrés
    v = ((target - target.mean()) ** 2).sum()  # somme des carrés totale
    return 1 - u / v


# ------------------- Main -------------------
def main():
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

    # Normalization
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

    theta_final = gradient_descent(X, y, theta, 0.01, axis, prediction_line[0], cost_line[0])

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

    score = r_squared(y, predictions) * 100
    print(f"⚡ Performance: {round(score, 2)}%")


if __name__ == "__main__":
    main()
