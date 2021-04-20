import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Functions
def get_regression_predictions(input_feature, intercept, slope):
    predicted_output = input_feature * slope + intercept
    return predicted_output


def simple_linear_regression(input_feature, output):
    # Computing sums needed to calculate slope and intercept
    regr = LinearRegression()
    regr.fit(input_feature, output)
    intercept = regr.intercept_
    slope = regr.coef_
    return (intercept, slope)


def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    y_hat = get_regression_predictions(input_feature, intercept, slope)
    RSS = (output - y_hat) ** 2
    return sum(RSS)


def inverse_regression_predictions(output, intercept, slope):
    estimated_input = (output - intercept) / slope
    return estimated_input


# Dictionary with the correct dtypes for the DataFrame columns
dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int,
              'sqft_living15': float, 'grade': int, 'yr_renovated': int,
              'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
              'sqft_lot15': float, 'sqft_living': float, 'floors': str,
              'condition': int, 'lat': float, 'date': str, 'sqft_basement': int,
              'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}
#
sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
train_data = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)
test_data = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)
# print(data["price"].shape)
# Split training and test
X_train, X_test, y_train, y_test = train_test_split(sales["sqft_living"], sales["price"], test_size=0.2, random_state=0)
# print(X_train.values.reshape(1, -1))
x_train_sqft = np.array(X_train).reshape(-1, 1)
x_train_price = np.array(y_train).reshape(-1, 1)
#
x_test_sqft = np.array(X_test).reshape(-1, 1)
y_test_price = np.array(y_test).reshape(-1, 1)
#
response = simple_linear_regression(x_train_sqft, x_train_price)
print(response[0][0])  # intercept
print(response[1][0][0])  # slope

print(get_regression_predictions(2650, response[0][0], response[1][0][0]))
# print(get_regression_predictions(x_train_sqft,response[0][0],response[1][0][0])-y_train)
# get resdual sum of square
# 1.201918e+15
print(get_residual_sum_of_squares(X_train, y_train, response[0][0], response[1][0][0]))
# get inverse
print(inverse_regression_predictions(800000, response[0][0], response[1][0][0]))
# number 13
#t Resdual sum of squares
print(get_residual_sum_of_squares(x_test_sqft, y_test_price, response[0][0], response[1][0][0]))#2.67770023e+14]
