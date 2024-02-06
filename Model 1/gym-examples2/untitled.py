import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Value to find
value_to_find = 5

# Find indices where value equals value_to_find
indices = np.where(matrix == value_to_find)