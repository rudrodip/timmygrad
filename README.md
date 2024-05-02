# Timmygrad

Timmygrad is a single value gradient descent optimizer for Python. It is designed to be simple and easy to use, with a focus on readability and understandability. It is not designed for performance, but rather for educational purposes.

<p align="center">
  <img src="https://github.com/rudrodip/timmygrad/blob/main/timmy.gif" />
</p>
<p align="center">this is timmy btw</p>

Here is a simple Linear Regression example using Timmygrad:

```python
m = Value(0.0)
c = Value(0.0)

alpha = 0.01 # learning rate
epochs = 200

for epoch in range(epochs):
  for x, y in zip(X, Y):
    # forward pass
    y_pred = m * x + c

    # compute loss
    loss = (y - y_pred) ** 2

    # backward pass
    loss.backward()

    # update weights
    m.data -= alpha * m.grad
    c.data -= alpha * c.grad

    # reset gradients
    m.grad = 0
    c.grad = 0
```

<p align="center">
  <img src="https://github.com/rudrodip/timmygrad/blob/main/linear_regression.png" />
</p>
