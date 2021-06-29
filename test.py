import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, max_error

# dummy data
n = 50
x = np.linspace(0, 10, n)
y = 2 * x - 1  # y=2x-1
y += np.random.normal(size=y.size)  # noise

data = pd.DataFrame([x, y], index=["X", "Y"]).transpose()

# split data
x_train, x_test, y_train, y_test = train_test_split(x, y)

# test training to random data
model = LinearRegression()
model.fit(x_train.reshape(-1, 1), y_train)
y_predict = x_test * model.coef_ + model.intercept_

# find metrics
metrics = pd.Series([r2_score(y_test, y_predict),
                     mean_squared_error(y_test, y_predict),
                     max_error(y_test, y_predict)],
                    ["R2", "Mean Squared Error", "Max Error"])

print(data.head(), model.coef_, model.intercept_, metrics, sep="\n")

# plot
fig, ax = plt.subplots()

ax.scatter(x, y, label="Test Data")
ax.plot(x_test, y_predict, c="r", label="Model Prediction")

ax.text(7, 1, f"R2: {round(metrics['R2'], 2)}", bbox=dict(boxstyle="square, pad=0.3", fc="white"))

ax.set_title("Test Training")
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.legend()

plt.show()
