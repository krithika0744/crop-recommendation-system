import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("Type of object loaded:", type(model))
print("Content:", model)
