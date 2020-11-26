import pickle
import sys
import numpy as np

if len(sys.argv) != 3:
	sys.exit("Error: incorrect number of arguments supplied. Usage: python predict.py model.sav input.csv")

try:
    model = pickle.load(open(sys.argv[1], 'rb'))
except:
    sys.exit("Error: model file not found. Try training with train.py first.")

try:
    X = np.genfromtxt(sys.argv[2],delimiter=',')
except:
    sys.exit("Error: input file not found.")


if X.ndim == 1:
	X = X.reshape(1, -1)

result = model.predict(X)
print(result)
