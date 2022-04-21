import pickle
import numpy as np

a = pickle.load(open("datasets/feature_vectors/huy.pkl", "rb"))
b = pickle.load(open("datasets/feature_vectors/long.pkl", "rb"))
c = pickle.load(open("datasets/feature_vectors/nhan.pkl", "rb"))

print(np.array_equal(a, b))
print(np.array_equal(a, c))
print(np.array_equal(c, b))