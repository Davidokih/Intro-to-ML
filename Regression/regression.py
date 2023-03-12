import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from numpy import sqrt


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# print(train.columns)

model = LinearRegression()

X_train = train.drop("SalePrice", axis=1)
Y_train = train.loc[:, "SalePrice"]

model.fit(X_train, Y_train)

X_test = test.drop("SalePrice", axis=1)
Y_test = test.loc[:, "SalePrice"]

prediction = model.predict(X_test)
comparison = pd.DataFrame({"Actual Value":Y_test, "Prediction": prediction})

# print(comparison.head())
# print(comparison.tail())

rmse = sqrt(mean_squared_error(Y_test, prediction))

# print(rmse)

correlations = train.corr()

# print(correlations)

salePrice_correlations = correlations["SalePrice"]

print(salePrice_correlations.sort_values(ascending=False).head(10))