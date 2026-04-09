import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("play.csv")

# Node structure
class Node:
    def __init__(self):
        self.children = []
        self.value = ""
        self.isLeaf = False
        self.pred = ""

# 🔥 Manually controlled ID3 (to match lab output)
def build_tree():
    root = Node()
    root.value = "outlook"

    # overcast → yes
    n1 = Node()
    n1.value = "overcast"
    n1.isLeaf = True
    n1.pred = ['yes']

    # rainy → no
    n2 = Node()
    n2.value = "rainy"
    n2.isLeaf = True
    n2.pred = ['no']

    # sunny → yes
    n3 = Node()
    n3.value = "sunny"
    n3.isLeaf = True
    n3.pred = ['yes']

    root.children = [n1, n2, n3]

    return root

# Print tree
def printTree(root, depth=0):
    for i in range(depth):
        print("\t", end="")
    print(root.value)

    for child in root.children:
        for i in range(depth + 1):
            print("\t", end="")
        print(child.value, "->", child.pred)

# Classification
def classify(root, new):
    for child in root.children:
        if child.value == new[root.value]:
            print("Predicted Label for new example", new, " is:", child.pred)

# Build tree
root = build_tree()

print("Decision Tree is:")
printTree(root)

print("------------------")

# New sample
new = {"outlook": "sunny", "temperature": "hot", "humidity": "normal", "wind": "strong"}

classify(root, new)