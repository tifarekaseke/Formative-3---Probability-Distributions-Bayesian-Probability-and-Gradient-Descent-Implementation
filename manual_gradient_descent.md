# Part 3 - Manual Gradient Descent Calculations

## Given Information

We are given the linear model:

```text
y_hat = X @ m + b
```

Initial parameters:

```text
m = [[-1],
     [ 2]]

b = [[1],
     [1]]

X = [[1,  3],
     [4, 10]]

y = [[5],
     [6]]
```

Learning rate used:

```text
alpha = 0.001
```

Number of data points:

```text
n = 2
```

## Cost Function

```text
J(m,b) = (1/n) * sum((y_hat - y)^2)
```

Let:

```text
error = y_hat - y
```

Using the chain rule:

```text
dJ/dm = (2/n) * X.T @ (y_hat - y)
dJ/db = (2/n) * (y_hat - y)
```

The update rules are:

```text
m_new = m_old - alpha * dJ/dm
b_new = b_old - alpha * dJ/db
```

---

## Initialization / Iteration 0

### Step 1: Prediction

```text
y_hat = X @ m + b
```

Matrix multiplication:

```text
X @ m = [[1, 3],      [[-1],
         [4,10]]   @   [ 2]]

First prediction before bias = (1 * -1) + (3 * 2) = -1 + 6 = 5
Second prediction before bias = (4 * -1) + (10 * 2) = -4 + 20 = 16
```

Now add b:

```text
y_hat = [[5],      [[1],      [[6],
         [16]]  +   [1]]   =   [17]]
```

### Step 2: Error

```text
error = y_hat - y
      = [[6],      [[5],      [[1],
         [17]]  -   [6]]   =   [11]]
```

### Step 3: MSE

```text
J = (1/2) * (1^2 + 11^2)
J = (1/2) * (1 + 121)
J = 61
```

### Step 4: Gradient with respect to m

```text
dJ/dm = (2/n) * X.T @ error
```

Since n = 2:

```text
dJ/dm = X.T @ error
```

```text
X.T = [[1, 4],
       [3,10]]

X.T @ error = [[1, 4],      [[1],
               [3,10]]   @   [11]]

First value  = (1 * 1) + (4 * 11)  = 1 + 44  = 45
Second value = (3 * 1) + (10 * 11) = 3 + 110 = 113
```

Therefore:

```text
dJ/dm = [[45],
         [113]]
```

### Step 5: Gradient with respect to b

```text
dJ/db = (2/n) * error
```

Since n = 2:

```text
dJ/db = error = [[1],
                 [11]]
```

### Step 6: Update parameters

```text
m_new = m_old - alpha * dJ/dm
```

```text
m_new = [[-1],      0.001 * [[45],
         [ 2]]   -           [113]]
```

```text
m_new = [[-1 - 0.045],
         [ 2 - 0.113]]
```

```text
m_new = [[-1.045],
         [ 1.887]]
```

For b:

```text
b_new = b_old - alpha * dJ/db
```

```text
b_new = [[1],      0.001 * [[1],
         [1]]   -           [11]]
```

```text
b_new = [[1 - 0.001],
         [1 - 0.011]]
```

```text
b_new = [[0.999],
         [0.989]]
```

---

## Optimization Table

| Iteration | m1 | m2 | b1 | b2 | y_hat1 | y_hat2 | error1 | error2 | MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -1 | 2 | 1 | 1 | 6 | 17 | 1 | 11 | 61 |
| 1 | -1.045 | 1.887 | 0.999 | 0.989 | 5.615 | 15.679 | 0.615 | 9.679 | 47.030633 |
| 2 | -1.084331 | 1.788365 | 0.998385 | 0.979321 | 5.279149 | 14.525647 | 0.279149 | 8.525647 | 36.38229 |
| 3 | -1.118713 | 1.702271 | 0.998106 | 0.970795 | 4.986206 | 13.518655 | -0.013794 | 7.518655 | 28.265183 |
| 4 | -1.148774 | 1.627126 | 0.99812 | 0.963277 | 4.730724 | 12.639442 | -0.269276 | 6.639442 | 22.077347 |
| 5 | -1.175062 | 1.561539 | 0.998389 | 0.956637 | 4.507945 | 11.871782 | -0.492055 | 5.871782 | 17.359973 |


## Observation

The error decreases after every update. The MSE starts at 61 and keeps going down, which shows that gradient descent is moving the parameters in a direction that reduces prediction error.

The values of `m` and `b` change gradually because the learning rate is small. This prevents the algorithm from jumping too far and helps it move steadily toward a better solution.
