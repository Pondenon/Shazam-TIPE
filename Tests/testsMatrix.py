from matrix import *
  
matrix1 = Matrix([[1, 2], [3, 4]])
matrix2 = Matrix([[5, 6], [7, 8]])
scalar = 2

# Matrix multiplication
product_matrix = matrix1 * matrix2
print("Product of matrices:\n", product_matrix.values)

# Scalar multiplication
scaled_matrix = matrix1 * scalar
print(scaled_matrix)
print("Matrix scaled by scalar:\n", scaled_matrix.values)

print(scaled_matrix.transpose())
print(scaled_matrix.trace())

a = matrix1 + matrix2
print("\na = ", a)

print("\na is square = ", a.verifySquare())

a.increaseCoefficients(18)
print(a)

a.crop(2, 1)
print(a)

a.crop(1, 1)
print(a)

a.crop(0, 2)
print(a)
