import numpy as np
from load_data import load_data
from plot import plot
from flatten import flatten

xtr, ytr = load_data()
plot(xtr,ytr)
xtr = flatten(xtr)
print np.shape(xtr)