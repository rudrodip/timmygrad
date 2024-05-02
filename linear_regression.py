from timmygrad.engine import Value
import matplotlib.pyplot as plt
import numpy as np

# dataset of random values
X = np.random.rand(200)
Y = 2 * X + 1 + np.random.randn(200) * 0.3

# Initialize weights
m = Value(0.0)
c = Value(0.0)

# Learning rate
alpha = 0.01

# Number of iterations
epochs = 200

for epoch in range(epochs):
  for x, y in zip(X, Y):
    y_pred = m * x + c
    loss = (y - y_pred) ** 2
    loss.backward()


    m.data -= alpha * m.grad
    c.data -= alpha * c.grad

    m.grad = 0
    c.grad = 0

    if epoch % 10 == 0:
      print(f'loss={loss.data}')

print(f"m: {m.data}, c: {c.data}")

plt.scatter(X, Y)
plt.plot(X, [m.data * x + c.data for x in X], 'r')
plt.show()