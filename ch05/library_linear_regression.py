from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# data
price = [0.98, 0.99, 1, 1.12, 1.35, 1.46, 1.59, 1.88, 1.91, 2]
sales = [254, 250, 230, 228, 220, 200, 180, 176, 160, 150]

# linear regression with Scikit Learn Model
linear_regression_model = LinearRegression()
price_format = [[a] for a in price]
sales_format = [[a] for a in sales]
linear_regression_model.fit(price_format, sales_format)
model_m = linear_regression_model.coef_[0][0]
model_b = linear_regression_model.intercept_[0]
print("Scikit Learn Regression Model")
print(f"Regression Equation: y = {model_m}x + {model_b}")

# predict a value
row = 5
predicted_model_quality = linear_regression_model.predict([price_format[row]])
print(
    f"For price value {price[row]}, the predicted sales value was {predicted_model_quality[0][0]} and the actual value was {sales[row]}")

# get the coefficient of determination
score = linear_regression_model.score(price_format, sales_format)
print(f"Coefficient of determination: {score}")

# plot best fit line
predictions = linear_regression_model.predict(price_format)
plt.scatter(price, sales)
plt.plot(price, predictions, linestyle="solid")
plt.xlabel("price")
plt.ylabel("sales")
plt.title("Price vs. Sales")
plt.show()
