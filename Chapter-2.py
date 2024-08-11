import numpy as np

inputs = np.array([1,2,3,4])

weight1 = np.array([0.2,-0.3,0.6,0.1])
weight2 = np.array([0.4,0.8,-0.1,-0.3])
weight3 = np.array([0.2,0.3,-0.6,0.1])


bias1 = 2
bias2 = 3
bias3 = 0.5

outputs = []
outputs.append(np.sum(np.multiply(inputs, weight1))+bias1)
outputs.append(np.sum(np.multiply(inputs, weight2))+bias2)
outputs.append(np.sum(np.multiply(inputs, weight3))+bias3)
print("dumb way")
print(outputs)
# ---------------------------------------------------------------

weights = [[0.2,-0.3,0.6,0.1],
           [0.4,0.8,-0.1,-0.3],
           [0.2,0.3,-0.6,0.1]]

biases = [2,3,0.5]

outputs = np.dot(weights, inputs) + biases
print("smarter way")
print(outputs)
# --------------------------------------------------------------

inputs = [[1,2,3,4],
          [2,5,-1,2],
          [-1.5,2.7,3.3,-0.8]]
weights = weights
biases = biases

outputs = np.dot(inputs, np.array(weights).T) + biases
print("using input batch")
print(outputs)

