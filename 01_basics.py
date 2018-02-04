import numpy as np
import matplotlib.pyplot as plt
from typing import List

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# our model for the forward pass
def forward(x: float, w: float) -> float:
    return x * w


# Loss function
def loss(y_predict: float, y: float):
    return np.square(y_predict - y)


w_list: List[float] = []
mse_list: List[float] = []

for i in np.arange(0.0, 4.1, 0.1):
    print("w=", i)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_predict_val = forward(x_val, i)
        lo = loss(y_predict_val, y_val)
        l_sum += lo
        print("\t", x_val, y_val, y_predict_val, lo)
    print("MSE=", l_sum / 3)
    w_list.append(i)
    mse_list.append(l_sum / 3)

plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()
