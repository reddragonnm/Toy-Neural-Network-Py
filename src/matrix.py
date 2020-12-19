import numpy as np
import jsonpickle

class Matrix:
    def __init__(self, rows, cols=None):
        if isinstance(rows, np.ndarray):
            self.data = rows
            self.rows, self.cols = rows.shape
        else:
            self.rows = rows
            self.cols = cols
            self.data = np.full((rows, cols), 0)

    def copy(self):
        return Matrix(np.copy(self.data))

    @staticmethod
    def fromArray(arr):
        return Matrix(np.array(arr))

    @staticmethod
    def subtract(a, b):
        return Matrix(a - b)

    def toArray(self, dtype=list):
        return dtype(self.data.flatten())

    def randomize(self):
        self.data = np.random.rand(self.rows, self.cols) * 2 - 1

    def add(self, n):
        if isinstance(n, Matrix):
            self.data += n.data
        else:
            self.data += n

    @staticmethod
    def transpose(matrix):
        if isinstance(matrix, Matrix):
            return Matrix(matrix.data.T)
        else:
            return Matrix(matrix.T)

    @staticmethod
    def multiply(a, b):
        return Matrix(np.dot(a.data, b.data))

    def multiply(self, n):
        self.data *= n

    def map_(self, func):
        np.vectorize(func)(self.data)

    def print_(self):
        for i in self.data:
            i = list(i)
            print(' '.join(map(str,i)))

    def serialize(self):
        return jsonpickle.encode(self)

    @staticmethod
    def deserialize(data):
        assert isinstance(data, str)
        return jsonpickle.decode(data)


if __name__ == '__main__':
    m = Matrix(3, 2)
    print(m.serialize())
    print(Matrix.deserialize(m.serialize()).data)
