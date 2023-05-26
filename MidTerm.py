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


def determinant_2x2_mat(a):
    det = a[0, 0]*a[1, 1] - a[0, 1]*a[1, 0]
    return det


def determinant_3x3_mat(a):
    det = a[0, 0]*(a[1, 1]*a[2, 2] - a[1, 2]*a[2, 1]) - a[0, 1]*(a[1, 0] *
                                                                 a[2, 2] - a[1, 2]*a[2, 0]) + a[0, 2]*(a[1, 0]*a[2, 1] - a[1, 1]*a[2, 0])
    return det


def inverse_matrix(a):
    det = determinant_2x2_mat(a)
    inv = np.array([[a[1, 1], -a[0, 1]], [-a[1, 0], a[0, 0]]])/det
    return inv


# Test and Print Identity:
A = np.array([[1, 2], [3, 4]])
InvA = inverse_matrix(A)
print(np.dot(InvA, A))


##########################
#       Question 2       #
##########################
"""
Using the functions above find the point of intersection to the nearest 1e-3 tolerance.
24x + 6y = 5
5x + 12y = 18
"""
# Solution Here (print solution):
A = np.array([[24, 6], [5, 12]])
B = np.array([5, 18])
x = np.dot(inverse_matrix(A), B)
print(x)

##########################
#       Question 3       #
##########################
"""
Implement a function for finding the eigenvalue and eigenvectors. 
You are not allowed to use a function that calculates the 
eigenvalue or eigenvector.

Hint: You may use a function to find the roots and the norm. 
"""


def find_eigens(a):
    eigenvalues = np.roots([1, -np.trace(a), determinant_2x2_mat(a)])
    eigenvector = np.zeros((2, 2))
    for i in range(2):
        eigenvector[:, i] = np.array(
            [eigenvalues[i] - a[1, 1], a[1, 0]])/a[0, 0]
        eigenvector[:, i] = eigenvector[:, i]/np.linalg.norm(eigenvector[:, i])
    return eigenvalues, eigenvector


# Test Cases:
print(find_eigens(np.array([[1, 2], [1, 1]])))
# Answer: [ 2.41421356, -0.41421356], [[0.81649658, 0.57735027], [-0.81649658,  0.57735027]]

print(find_eigens(np.array([[16, 2], [4, 13]])))
# Answer: [17.70156212, 11.29843788], [[0.76164568, 0.64799372], [-0.39144501,  0.9202015 ]]
