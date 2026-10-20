# Data
price = [0.98, 0.99, 1, 1.12, 1.35, 1.46, 1.59, 1.88, 1.91, 2]
sales = [254, 250, 230, 228, 220, 200, 180, 176, 160, 150]

# manual regression
n = len(price)
sum_x = sum(price)
sum_y = sum(sales)
x_mean = sum_x / n
y_mean = sum_y / n
sum_x_times_y = sum([a*b for a, b in zip(price, sales)])
sum_x_squareds = sum([a*a for a in price])
print(f"n: {n}")
print(f"Sum of x values {sum_x}")
print(f"Sum of y values {sum_y}")
print(f"x mean value {x_mean}")
print(f"y mean value {y_mean}")
print(f"Sum of x times y {sum_x_times_y}")
print(f"Sum of squared x values {sum_x_squareds}")

# calculate numerator and denominator for m
Sxy = sum_x_times_y - (sum_x*sum_y)/n
Sxx = sum_x_squareds - (sum_x*sum_x)/n
print(f"Sxy: {Sxy} Sxx: {Sxx}")

# get line variables
m = Sxy/Sxx
b = y_mean - m*x_mean
print(f"Regression Equation: y = {m}x + {b}")

# predict a value
row = 5
price_val = price[row]
sales_val = sales[row]
predicated_sales = price_val*m + b
print(
    f"For price value {price_val}, the predicted sales value was {predicated_sales} and the actual value was {sales_val}")

# coefficent of determination
predictions = [a*m + b for a in price]
residual_squares = sum([(a-b)*(a-b) for a, b in zip(sales, predictions)])
total_squares = sum([(a-y_mean)*(a-y_mean) for a in sales])
r_squared = 1 - residual_squares/total_squares
print(f"Coefficient of determination: {r_squared}")
