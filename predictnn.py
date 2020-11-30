import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

try:
    model = keras.models.load_model("nnmodel")
except:
    sys.exit("Error: model folder not found. Try training with catnn.py first.")

try:
    X = np.genfromtxt('test.csv',delimiter=',')[1:]
except:
    sys.exit("Error: input file not found.")

result = model.predict_classes(X)
print(result)
