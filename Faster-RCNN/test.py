import data.dataset as dataset
import numpy as np

aa = np.array([[1,2,3, 4], [2,3, 4, 5]])

# aa[:, 2::4] = np.array([[10], [10]])


print(aa[:,slice(1, 4, 2)])

