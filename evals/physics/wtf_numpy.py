import numpy as np


x = np.random.randn(100000, 10000)
y = np.random.randn(10000, 10000)

z = x @ y

# %%
q = x + x
