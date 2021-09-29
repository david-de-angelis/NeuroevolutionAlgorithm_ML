import numpy as np 
import math
from random import random

#Standard Matrix Functions to support operations of a neural network
class Matrix(object):
    def __init__(self, _num_rows, _num_cols):
        self.num_rows = _num_rows
        self.num_cols = _num_cols
        self.grid = np.zeros((self.num_rows, self.num_cols)).tolist()
    
    # Used to print the current value of the grid, as numpy formats it for output, super useful for debugging
    def __str__(self):
        return str(np.array(self.grid))

    # Used to apply some function over each cell of the matrix.
    def map(self, func):
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                self.grid[i][j] = func(self.grid[i][j], i, j)

    # Used to randomise each cell of the matrix with a value between -1 & 1
    def randomise(self):
        self.map(lambda x, i, j: random() * 2.0 - 1.0)

    # Matrix Scalar Addition Operation
    def add(self, num):
        self.map(lambda x, i, j: x + num)

    # Matrix Element-Wise Addition Operation
    def addMatrix(self, otherMatrix):
        self.map(lambda x, i, j: x + otherMatrix.grid[i][j])

    # Matrix Scalar Multiplication Operation
    def multiplyBy(self, num):
        self.map(lambda x, i, j: x * num)
    
    # Matrix Element-Wise Multiplication (Hadamard / Schur Product) Operation
    def multiplyByMatrix(self, otherMatrix):
        self.map(lambda x, i, j: x * otherMatrix.grid[i][j])

    # Normalize the value to be between 0 and 1
    def sigmoidise(self):
        self.map(lambda x, i, j: 1 / (1 + math.exp(-x/10.0)))

    def copy(self):
        result = Matrix.mapToNew(self, lambda x, i, j: x)
        return result

    # Condense into a 1D array, to return output
    def toArray(self):
        result = []
        for rows in range(self.num_rows):
            for cols in range(self.num_cols):
                result.append(self.grid[rows][cols])
        return result

    ##################### STATIC METHODS  #####################
    
    @staticmethod
    def mapToNew(matrix, func):
        result = Matrix(matrix.num_rows, matrix.num_cols)
        for i in range(matrix.num_rows):
            for j in range(matrix.num_cols):
                result.grid[i][j] = func(matrix.grid[i][j], i, j)
        return result

    #Subtract one matrix by another
    @staticmethod
    def subtractMatricies(matrixA, matrixB):
        return Matrix.mapToNew(matrixA, lambda x, i, j: x - matrixB.grid[i][j])

    #Transpose the matrix (Turn each row into a column)
    @staticmethod
    def transposeMatrix(matrix):
        result = Matrix(matrix.num_cols, matrix.num_rows) #e.g. change from 3,2 matrix to 2,3 matrix
        result.map(lambda x, i, j: matrix.grid[j][i])
        return result

    #Matrix Dot Product
    @staticmethod
    def multiplyMatricies(matrixA, matrixB):
        if matrixA.num_cols != matrixB.num_rows:
            print("Unable to perform Matrix Dot Product - As Matrix A does not have the same amount of rows that Matrix B has columns")
            return
        
        result = Matrix(matrixA.num_rows, matrixB.num_cols)
        result.grid = np.dot(matrixA.grid, matrixB.grid)
        return result
    
    @staticmethod
    def createFromArray(input_array):
        result = Matrix(len(input_array), 1) #This is only used for the input, so a column count of 1 is fine.
        for row in range(result.num_rows):
            result.grid[row][0] = input_array[row]
        return result