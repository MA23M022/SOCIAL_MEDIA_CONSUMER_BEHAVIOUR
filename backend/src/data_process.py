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


# Save the preprocessed data into csv file

removed_outliers_data.to_csv(r'C:\Users\soura\Desktop\Resume_Projects\SOCIAL_MEDIA_PROJECT\data\preprocessed\cleaned_data.csv', index=False)

print(f"Data preprocessing has been done")
