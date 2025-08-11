# --------------------------------------- Import libraries  ------------------------------------------------
print("-------------------------------Import essential libraries --------------------------------------------")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns



print("-------------------------------Read dataset and Extract relevent columns from dataset -------------------------------")
# Load the dataset
raw_df = pd.read_csv(r'C:\Users\soura\Desktop\Resume_Projects\SOCIAL_MEDIA_PROJECT\data\raw\data.csv')
df = raw_df.copy()

# Remove rows with missing data
cleaned_df = df.dropna()

# Droped the irrelevent column 'url'
droped_df = cleaned_df.drop(columns= 'Url')

droped_df['Date Sampled'] = pd.to_datetime(droped_df['Date Sampled'])

droped_df['Month_Sampled'] = droped_df['Date Sampled'].dt.month

droped_df = droped_df.drop(columns = 'Date Sampled', axis=1)

droped_df = droped_df.iloc[:, [0, 1, 3, 2]]



#------------------------------------------- Extract the facebook datapoints ----------------------------------

print("--------------------------------------- Extract out the facebook datapoints ---------------------------------------")
# Load the cleaned dataset
mo_data = droped_df.copy()

# Filter data for Facebook
facebook_data = (mo_data[mo_data['Platform'].isin(['Facebook'])]).drop(columns = 'Platform')

facebook_agencies = facebook_data['Agency'].unique()
print('The agencies whoever shows items through the facebook platform:\n')
print(facebook_agencies)



#------------------------------------------ Extract the twitter datapoints --------------------------------------

print("--------------------------------------- Extract out the twitter datapoints ---------------------------------------")
mo_data = droped_df.copy()

# Filter data for Twitter
twitter_data = (mo_data[mo_data['Platform'].isin(['Twitter'])]).drop(columns = 'Platform')

twitter_agencies = twitter_data['Agency'].unique()
print('The agencies whoever shows items through the twitter platform:\n')
print(twitter_agencies)




#------------- Create a dataframe that has columns ['Agency', 'Month_Sampled', 'fb_data', 'tw_data'] ----------------------

print("-------------------------------------- Create the relevent dataset -----------------------------------------")
agency_list = []
for ele in facebook_agencies:
    if ele in twitter_agencies:
        agency_list.append(ele)

print('The agencies whoever shows the items in both facebook and twitter platforms: \n')
print(agency_list)
print(f'\n The length of that agency list is : {len(agency_list)}')


fb_df = facebook_data.copy()
tw_df = twitter_data.copy()

data_sheet = []
value = 0
for agency in agency_list:
    value += 1
    fb_dummy = (fb_df[fb_df['Agency'].isin([agency])]).drop(columns = 'Agency')
    tw_dummy = (tw_df[tw_df['Agency'].isin([agency])]).drop(columns = 'Agency')

    fb_dummy_list = sorted((fb_dummy.values).tolist())             # sorted with respect to month.
    tw_dummy_list = sorted((tw_dummy.values).tolist())

    while(fb_dummy_list and tw_dummy_list):
        lst = [agency]
        val = 0
        if(fb_dummy_list[0][0] == tw_dummy_list[0][0]):
            dummy_lst = [tw_dummy_list[0][0], fb_dummy_list[0][1], tw_dummy_list[0][1]]
            lst.extend(dummy_lst)
            data_sheet.append(lst)
            val = 1
        elif(fb_dummy_list[0][0] > tw_dummy_list[0][0]):
            val = 2
        else:
            val = 3

        if(val == 1):
            fb_dummy_list.pop(0)
            tw_dummy_list.pop(0)
        if(val == 2):
            tw_dummy_list.pop(0)
        if(val == 3):
            fb_dummy_list.pop(0)


print(f'Number of datas presents in the list is : {len(data_sheet)}')


merged_facebook_twitter_data = pd.DataFrame(data_sheet, columns = ['Agency', 'Month_Sampled', 'fb_data', 'tw_data'])

convert_dict = {'Month_Sampled': int}
merged_facebook_twitter_data = merged_facebook_twitter_data.astype(convert_dict)



# Use the log transformation to overcome right skewed data
merged_facebook_twitter_data["log_fb_data"] = np.log1p(merged_facebook_twitter_data["fb_data"])

merged_facebook_twitter_data["log_tw_data"] = np.log1p(merged_facebook_twitter_data["tw_data"])

merged_facebook_twitter_data = merged_facebook_twitter_data[["Agency", "Month_Sampled", "log_fb_data", "log_tw_data"]]




# --------------------------------------------clean outlier datapoints-----------------------------------------

print("----------------------------------- Clean outliers from the dataset ------------------------------------------")
removed_outliers_data = merged_facebook_twitter_data.copy()

# For 'feature_name_log'
Q1_feature = removed_outliers_data['log_fb_data'].quantile(0.25)
Q3_feature = removed_outliers_data['log_fb_data'].quantile(0.75)
IQR_feature = Q3_feature - Q1_feature
lower_bound_feature = Q1_feature - (1.5 * IQR_feature)
upper_bound_feature = Q3_feature + (1.5 * IQR_feature)

# For 'target_variable_log'
Q1_target = removed_outliers_data['log_tw_data'].quantile(0.25)
Q3_target = removed_outliers_data['log_tw_data'].quantile(0.75)
IQR_target = Q3_target - Q1_target
lower_bound_target = Q1_target - (1.5 * IQR_target)
upper_bound_target = Q3_target + (1.5 * IQR_target)

print(f"Feature: Lower Bound = {lower_bound_feature:.2f}, Upper Bound = {upper_bound_feature:.2f}")
print(f"Target: Lower Bound = {lower_bound_target:.2f}, Upper Bound = {upper_bound_target:.2f}")


# Handling outliers (capping example):
removed_outliers_data['log_fb_data'] = np.where(removed_outliers_data['log_fb_data'] > upper_bound_feature,
                                                upper_bound_feature,
                                                np.where(removed_outliers_data['log_fb_data'] < lower_bound_feature,
                                                        lower_bound_feature,
                                                        removed_outliers_data['log_fb_data']))

removed_outliers_data['log_tw_data'] = np.where(removed_outliers_data['log_tw_data'] > upper_bound_target,
                                                upper_bound_target,
                                                np.where(removed_outliers_data['log_tw_data'] < lower_bound_target,
                                                        lower_bound_target,
                                                        removed_outliers_data['log_tw_data']))





# ----------------------------------------- Linear and polynominal regression -----------------------------------------

print("-------------------------------------- Linear and Polynominal regression model -----------------------------------")
X = removed_outliers_data[['log_fb_data']].values
y = removed_outliers_data[['log_tw_data']].values

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.15, random_state=42)

print(f'X-Train data shape : {X_train1.shape}, y-train data shape : {y_train1.shape}' )
print(f'X-test data shape : {X_test1.shape}, y-test data shape : {y_test1.shape}')


# Create polynomial features matrix
def create_polynomial_matrix(X, degree):
    X_poly_mat = []

    for i in range(1, degree + 1):
        X_poly = np.ones((len(X), 1))
        for d in range(1, i + 1):
            X_poly = np.concatenate((X_poly, X**d), axis=1)
        X_poly_mat.append(X_poly)
    return X_poly_mat


# Define degree of polynomial
degree = 11

X_train_poly_mat = create_polynomial_matrix(X_train1, degree)


# computing the weight matrix for each polynominal.
theta_mat = []
for ele in X_train_poly_mat:  # Solve for coefficients using normal equation (closed-form)
    theta = np.linalg.inv(ele.T.dot(ele)).dot(ele.T).dot(y_train1)
    theta_mat.append(theta)

# printing the weights for each polynominal.
k = 1
for ele in theta_mat:
    print(f'The weight matrix for polynominal regression of degree {k} is :')
    print(ele)
    k += 1




# Make predictions
X_test_poly_mat = create_polynomial_matrix(X_test1, degree)

def predict(X_poly_mat, theta_mat):
    y_pred_mat = []
    for ele1 , ele2 in zip(X_poly_mat, theta_mat):
        y_pred = ele1.dot(ele2)
        y_pred_mat.append(np.expm1(y_pred))
    return y_pred_mat

y_pred_mat = predict(X_test_poly_mat, theta_mat)



# Evaluate the model
def mean_squared_error(y_true, y_pred_mat):
    mean_list = []
    for ele in y_pred_mat:
        value = np.mean((y_true - ele)**2)
        mean_list.append(value)
    return mean_list




def _r2_score(y_true, y_pred_mat):
    r2_list = []
    denominator = np.sum((y_true - np.mean(y_true))**2)
    for ele in y_pred_mat:
        numerator = np.sum((y_true - ele)**2)
        val = 1 - (numerator / denominator)
        r2_list.append(val)
    return r2_list



mse_list = mean_squared_error(np.expm1(y_test1), y_pred_mat)
r2_list = _r2_score(np.expm1(y_test1), y_pred_mat)


index = 1
for ele1, ele2 in zip(mse_list, r2_list):
    print(f"Root mean Squared Error for {index} degree polynominal model : {pow(ele1, 0.5)}")
    print(f"R2 score for {index} degree polynominal model : {ele2}")
    index += 1





#  --------------------------------------- Tree based models-------------------------------------------

print("-------------------------------------------Tree based metihds-------------------------------------------------")
dummy_X = removed_outliers_data[['Agency', 'Month_Sampled']]

dummy_y = removed_outliers_data[['log_tw_data']]

one_hot_encoded_data = pd.get_dummies(dummy_X, columns = ['Agency', 'Month_Sampled'], dtype = int)

one_hot_encoded_data = one_hot_encoded_data.drop(columns=["Agency_DCA", "Month_Sampled_12"], axis=1) #Avoid multi-colinearity.

one_hot_encoded_data['log_fb_data'] = removed_outliers_data['log_fb_data']

X = one_hot_encoded_data

Y = removed_outliers_data['log_tw_data']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, Y, test_size=0.15, random_state=42)
print(f'X-Train data shape : {X_train2.shape}, y-train data shape : {y_train2.shape}' )
print(f'X-test data shape : {X_test2.shape}, y-test data shape : {y_test2.shape}')



print("--------------------------------------------- Decision Tree method ----------------------------------------------")
# create a regressor object
regressor = DecisionTreeRegressor(random_state = 0)

# fit the regressor with X and Y data
regressor.fit(X_train2, y_train2)

y_pred = regressor.predict(X_test2)

log_rsme = root_mean_squared_error(y_test2, y_pred)
print(f"The root mean square score for Decision Tree (log transform) is {log_rsme}")

log_r2_score_val = r2_score(y_test2, y_pred)
print(f"The R2 score for Decision Tree (log transform) is {log_r2_score_val}")


rsme = root_mean_squared_error(np.expm1(y_test2), np.expm1(y_pred))
print(f"The root mean square score for Decision Tree is {rsme}")

r2_score_val = r2_score(np.expm1(y_test2), np.expm1(y_pred))
print(f"The R2 score for Decision Tree is {r2_score_val}")



print("------------------------------------------ Random Forest method --------------------------------------------------")
# Fitting Random Forest Regression to the dataset
forest_regressor = RandomForestRegressor(n_estimators=100, random_state=0, oob_score=True)

# Fit the regressor with x and y data
forest_regressor.fit(X_train2, y_train2)

predictions = forest_regressor.predict(X_test2)

log_rsme = root_mean_squared_error(y_test2, predictions)
print(f"The root mean square score for Random Forest (log transform) is {log_rsme}")

log_r2_score_val = r2_score(y_test2, predictions)
print(f"The R2 score for Random Forest (log transform) is {log_r2_score_val}")

rsme = root_mean_squared_error(np.expm1(y_test2), np.expm1(predictions))
print(f"The root mean square score for Random Forest is {rsme}")

r2_score_val = r2_score(np.expm1(y_test2), np.expm1(predictions))
print(f"The R2 score for Random Forest is {r2_score_val}")




print("----------------------------------------------- Xgboost method ---------------------------------------------------------")
# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Initialize the XGBRegressor model
model = XGBRegressor()

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train2, y_train2)


# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Make predictions on test set using the best model
best_model = grid_search.best_estimator_
predictions2 = best_model.predict(X_test2)

#predictions2 = predictions2.reshape(-1, 1)

log_rsme = root_mean_squared_error(y_test2, predictions2)
print(f"The root mean square score for Xgboost (log transform) is {log_rsme}")

log_r2_score_val = r2_score(y_test2, predictions2)
print(f"The R2 score for Xgboost (log transfrom) is {log_r2_score_val}")

rsme = root_mean_squared_error(np.expm1(y_test2), np.expm1(predictions2))
print(f"The root mean square score for Xgboost is {rsme}")

r2_score_val = r2_score(np.expm1(y_test2), np.expm1(predictions2))
print(f"The R2 score for Xgboost is {r2_score_val}")

