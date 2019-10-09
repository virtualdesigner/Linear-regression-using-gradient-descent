from numpy import *


def squared_error(m, b, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += ((y - (m * x + b)) ** 2)
    return total_error / len(points)


def gradient_descent(m, b, points, learning_rate):
    partial_derivative_with_respect_to_b = 0
    partial_derivative_with_respect_to_m = 0
    N = len(points)
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        partial_derivative_with_respect_to_b += -(2/N)*(y - (m * x + b))
        partial_derivative_with_respect_to_m += -(2/N)*(x*(y - (m * x + b)))
    new_m = m - (partial_derivative_with_respect_to_m * learning_rate)
    new_b = b - (partial_derivative_with_respect_to_b * learning_rate)
    return [new_m, new_b]


def linear_regression_using_gradient_descent(m, b, points, learning_rate, no_of_iterations):
        # print(points)
    print("Initial value of 'm' :", m, "Initial value of 'b' :",
          b, "Initial error :", squared_error(m, b, points))
    for i in range(0, no_of_iterations):
        [m, b] = gradient_descent(m, b, points, learning_rate)
    print("Value of 'm' after regression : ", m, "Value of 'b' after regression : ",
          b, "Error after regression: ", squared_error(m, b, points))


if __name__ == '__main__':
    linear_regression_using_gradient_descent(
        0, 0, genfromtxt('data.csv', delimiter=","), 0.0001, 10000)
