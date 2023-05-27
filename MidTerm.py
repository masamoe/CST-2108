import numpy as np
import matplotlib.pyplot as plt

##########################
#       Question 1       #
##########################
"""
a) Write a function for finding the determinant of a 2x2 matrix.
b) Write a function for finding the determinant of a 3x3 matrix.
c) Write a function for finding the inverse 2x2 matrix.
d) Verify you get the Identity matrix InvA*A.
You must implemented it yourself.
"""


def determinant_2x2_mat(matrix):
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]

    determinant = (a * d) - (b * c)

    return determinant


def determinant_3x3_mat(matrix):
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[0][2]
    d = matrix[1][0]
    e = matrix[1][1]
    f = matrix[1][2]
    g = matrix[2][0]
    h = matrix[2][1]
    i = matrix[2][2]

    determinant = (a * e * i) + (b * f * g) + (c * d * h) - \
        (c * e * g) - (b * d * i) - (a * f * h)

    return determinant


def inverse_matrix(matrix):
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]

    determinant = (a * d) - (b * c)
    inverse_matrix = [[d, -b], [-c, a]]
    scalar = 1 / determinant
    for i in range(2):
        for j in range(2):
            inverse_matrix[i][j] *= scalar

    return inverse_matrix


# Test and Print Identity:
def verify_inverse(matrix):
    size = matrix.shape[0]

    identity = np.eye(size)
    inverse = np.linalg.inv(matrix)
    result = np.dot(inverse, matrix)
    is_identity = np.allclose(result, identity)

    return is_identity


##########################
#       Question 2       #
##########################
"""
Using the functions above find the point of intersection to the nearest 1e-3 tolerance.
24x + 6y = 5
5x + 12y = 18
"""
# Solution Here (print solution):

coefficient = np.array([[24, 6], [5, 12]])
constant = np.array([5, 18])

inverse = np.array(inverse_matrix(coefficient))
intersection_point = np.dot(inverse, constant)

print(tuple(intersection_point))
# Answer: (-0.186, 1.577)

##########################
#       Question 3       #
##########################
"""
Implement a function for finding the eigenvalue and eigenvectors. 
You are not allowed to use a function that calculates the 
eigenvalue or eigenvector.

Hint: You may use a function to find the roots and the norm. 
"""


def find_eigens(matrix):
    eigenvalues = []
    eigenvectors = []

    def calculate(matrix):
        size = len(matrix)
        if size == 2:
            a = matrix[0][0]
            b = matrix[0][1]
            c = matrix[1][0]
            d = matrix[1][1]

            eigenvalue1 = (a + d + ((a + d) ** 2 - 4 *
                           (a * d - b * c)) ** 0.5) / 2
            eigenvalue2 = (a + d - ((a + d) ** 2 - 4 *
                           (a * d - b * c)) ** 0.5) / 2

            eigenvalues.append(eigenvalue1)
            eigenvalues.append(eigenvalue2)

            eigenvector1 = [1, (eigenvalue1 - a) / b]
            eigenvector2 = [1, (eigenvalue2 - a) / b]

            eigenvectors.append(eigenvector1)
            eigenvectors.append(eigenvector2)

        else:
            for i in range(size):
                submatrix = matrix[:i] + matrix[i + 1:]
                calculate(submatrix)

    calculate(matrix)

    return eigenvalues, eigenvectors


# Test Cases:
print(find_eigens(np.array([[1, 2], [1, 1]])))
# Answer: [ 2.41421356, -0.41421356], [ 0.81649658, -0.81649658], [ 0.57735027,  0.57735027]]

print(find_eigens(np.array([[16, 2], [4, 13]])))
# Answer: [17.70156212, 11.29843788], [ 0.76164568, -0.39144501], [ 0.64799372,  0.9202015 ]]
