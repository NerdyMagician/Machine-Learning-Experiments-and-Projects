import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1,
return_X_y=True)
X = X.values

# print(X.shape)      # should be (70000, 784)
# print(X.head())     # should show real pixel numbers, not zero
# non_zero_rows = np.where(X.values.sum(axis=1) != 0)[0]

# if len(non_zero_rows) == 0:
#     print("All rows contain only zeros — dataset likely not loaded correctly.")
# else:
#     first_index = non_zero_rows[0]
#     print(f"First non-zero row index: {first_index}")
#     print(X.iloc[first_index])
y = y.astype(int).values
X = ((X/255.)-5)*2

import matplotlib.pyplot as plt
fig , ax = plt.subplots(nrows = 2, ncols =5 , sharex=True , sharey= False)
ax = ax.flatten()
for i in range(10):
    img = X[y == i][0].reshape(28,28)
    ax[i].imshow(img , cmap = 'Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# plt.show()
fig, ax = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(25):
    img = X[y == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
plt.tight_layout()
plt.show()
from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(X,y,test_size=10000,random_state=123, stratify=y)
X_train, X_valid, y_train, y_valid= train_test_split(X_temp, y_temp, test_size = 5000, random_state=123, stratify=y_temp)

from neuralnet import NeuralNetMLP
model = NeuralNetMLP(num_features = 28*28, num_hidden = 50, num_classes = 10)
print(model)







