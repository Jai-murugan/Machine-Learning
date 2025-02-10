# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score

# Import the data
car_details = pd.read_csv(r"J:\Data Science\Projects\Linear Regression\Car Details\CAR DETAILS FROM CAR DEKHO.csv")

# Understand the data
car_details.info()
car_details.describe()
car_details.isna().sum()

car_details.shape
car_details.duplicated().sum()
car_details[car_details.duplicated()]

# Removing duplicates to avoid issues in model training
car_df = car_details.drop_duplicates()
car_df.duplicated().sum()
car_df.isna().sum()

print(car_df["fuel"].unique())
print(car_df["seller_type"].unique())
print(car_df["transmission"].unique())
print(car_df["owner"].unique())


# Exploratory Data Analysis (EDA)
car_df["Brand"] = car_df['name'].str.split().str[0]
car_df.head()
car_df["Brand"].value_counts() 
# Maruti is the most resold car
car_df["fuel"].value_counts()
# Most cars are diesel or petrol vehicles
car_df["owner"].value_counts()
# Major sellers are First owner

price = car_df.groupby("Brand")["selling_price"].mean()
price.sort_values(ascending=False, inplace=True)
price = price.reset_index()
price['selling_price'] = price['selling_price'].astype(int)
print(price)

sns.barplot(data = price, x = "selling_price", y = "Brand", color = "dimgray", dodge = True)
plt.xlabel("Selling price in million")
plt.tight_layout()
plt.show()

# Dropping electric car since it has only one data, it give issues in model training
car_df.drop(car_df[car_df['fuel'] == 'Electric'].index, inplace=True)


# Feature Scaling
car_df["year"] = 2020 - car_df["year"]
car_df.rename(columns = {"year" : "age_of_the_car"}, inplace = True)

scale_standard = StandardScaler()
scaled_data = scale_standard.fit_transform(car_df[["age_of_the_car", "selling_price", "km_driven"]])
car_df_standardized = pd.DataFrame(scaled_data, columns = ["age_of_the_car", "selling_price", "km_driven"])
car_df.drop(["selling_price", "age_of_the_car", "km_driven", "name", "Brand"], axis = 1, inplace = True)


car_df.reset_index(drop=True, inplace=True)
car_df = pd.concat([car_df, car_df_standardized], axis = 1 )

# Split our data

X = car_df.drop(["selling_price"], axis = 1)
y = car_df["selling_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)

# Categorical Variables
categorical_vars = ["fuel", "seller_type", "transmission", "owner"]

one_hot_encoder = OneHotEncoder(sparse_output = False, drop = "first")  

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop = True), X_train_encoded.reset_index(drop = True)], axis = 1) 
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop = True), X_test_encoded.reset_index(drop = True)], axis = 1) 
X_test.drop(categorical_vars, axis = 1, inplace = True)

# Feature Selection

regressor = LinearRegression()
feature_selector = RFECV(regressor)

fit = feature_selector.fit(X_train,y_train)

optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features : {optimal_feature_count}")

X_train = X_train.loc[:, feature_selector.get_support()]  # This code is to select only object that is selected by feature selection algorithm
X_test = X_test.loc[:, feature_selector.get_support()]

plt.plot(range(1, len(fit.cv_results_['mean_test_score']) +1), fit.cv_results_['mean_test_score'], marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()

# Model Training

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict on the test

y_pred = regressor.predict(X_test)

# calculate R-Squared

r_squared = r2_score(y_test, y_pred)
print(r_squared)

# Cross validation

cv = KFold(n_splits = 4, shuffle= True, random_state= 42)       # Updating no of folds and enabling shuffle
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean() 

# Calculate Adjusted R-Squared

num_data_points, num_input_vars = X_test.shape

adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

# Extract Model Coefficient

coefficients = pd.DataFrame(regressor.coef_)
input_variable_names = pd.DataFrame(X_train.columns)
summary_stats = pd.concat([input_variable_names, coefficients], axis = 1)
summary_stats.columns = ["input_variable", "coefficient"]

#######################################################
# Decision Tree
######################################################

car_df.drop(["name", "Brand"], axis = 1, inplace = True)

# Split our data

X = car_df.drop(["selling_price"], axis = 1)
y = car_df["selling_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)

# Categorical Variables
categorical_vars = ["fuel", "seller_type", "transmission", "owner"]

one_hot_encoder = OneHotEncoder(sparse_output = False, drop = "first")  

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop = True), X_train_encoded.reset_index(drop = True)], axis = 1) 
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop = True), X_test_encoded.reset_index(drop = True)], axis = 1) 
X_test.drop(categorical_vars, axis = 1, inplace = True)

# Model Training
regressor_tree = DecisionTreeRegressor(random_state = 42, max_depth = 8)
regressor_tree.fit(X_train, y_train)

# Predict on the test

y_pred = regressor_tree.predict(X_test)

# calculate R-Squared

r_squared = r2_score(y_test, y_pred)
print(r_squared)

# Cross validation

cv = KFold(n_splits = 4, shuffle= True, random_state= 42)       # Updating no of folds and enabling shuffle
cv_scores = cross_val_score(regressor_tree, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean() 

# Calculate Adjusted R-Squared

num_data_points, num_input_vars = X_test.shape

adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

# A demostration of Overfitting

y_pred_training = regressor_tree.predict(X_train)
r2_score(y_train, y_pred_training)

# Finding the max_depth

max_depth_list = list(range(1,9))
accuracy_scores = []

for depth in max_depth_list:
    regressor_tree = DecisionTreeRegressor(max_depth = depth, random_state = 42)
    regressor_tree.fit(X_train, y_train)
    y_pred = regressor_tree.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    
max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_depth = max_depth_list[max_accuracy_idx]

# Plot of Max depth

plt.plot(max_depth_list, accuracy_scores)
plt.scatter(optimal_depth, max_accuracy, marker = "x", color = "red")
plt.title(f"Accuracy by Max Depth \n Optimal Tree depth : {optimal_depth} (Accuracy : {round(max_accuracy,4)})")
plt.xlabel("Max depth by Decision tree")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()

































