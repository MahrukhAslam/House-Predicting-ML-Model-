import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


data = pd.read_csv('perth_data.csv')
# Removes the rows that contains NULL values
data = data.dropna(axis=1)

y = data['PRICE']
features = ['LATITUDE', 'LONGITUDE','BEDROOMS','BATHROOMS','LAND_AREA']
X = data[features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create a machine learning model using a RandomForestRegressor algorithm
model = RandomForestRegressor(random_state=1)
model.fit(X_train, y_train)

# Evaluate the model's performance using the testing set:
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R2 Score:', r2)

# Visualize the performance of the model using a scatter plot
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted House Prices')
plt.show()
