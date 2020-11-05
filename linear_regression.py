#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The following project predicts the number of new cases
for covid-19 in the United States and around the world. 
It looks at data obtained for 3 months (March - May), the
compares it the next 3 months (June-August). 


Data was obtained from this website: 
https://ourworldindata.org/coronavirus-source-data
Courtesy from Our 'Our World in Data'

Modules you need to install: 
'''
pip install scikit-learn
'''

"""


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Import csv file using 
csv_file_path = './owid-covid-data.csv'
file_data = pd.read_csv(csv_file_path, parse_dates=['date'])   


# Columns of interest
columns = ['total_cases', 'new_cases', 'date']


start_date = '2020-03-01'
end_date = '2020-05-31'

target_end_date = '2020-08-31'


#################################################
# Section 1. Obtaining linear regression line for 
#            total number of cases worldwide
#################################################
iso_code = 'OWID_WRL'


# Converting 'date' column to appropriate datatype
file_data['date']=pd.to_datetime(file_data['date'])
file_data.date=pd.to_datetime(file_data.date)
#print(file_data.dtypes)
#print(file_data.date)


# Filters out dates, keeps only the dates of interest (not used)
start_date_time = pd.to_datetime(start_date)
end_date_time = pd.to_datetime(end_date)
filtered_chart = file_data.loc[(file_data['date'] > start_date_time) & (file_data['date'] < end_date_time)]
#print(filtered_chart.dtypes)
#print(filtered_chart[columns])


# filters by continent and date range
filtered_chart = file_data.loc[(file_data['iso_code'] == iso_code) & (file_data['date'] > start_date) & (file_data['date'] < end_date)]
#print(filtered_chart)
#print("The number of cases around the world for the dates " + str(start_date) + " through " +  str(end_date))
#print(filtered_chart[columns])



# Store the columns into an array
total_cases = np.array(filtered_chart['total_cases'])
new_cases = np.array(filtered_chart['new_cases'])
date_timeline = np.array(filtered_chart['date']) 

count = 0
index = []
for i in date_timeline: 
    count += 1
    
for i in range(0, count): 
    index.append(i)
    

# Plot cases worldwide
plt.title('Total cases worldwide')
plt.ylabel("Total number of cases")
plt.xlabel("Days since start date")
plt.scatter(index, total_cases, label = 'views')
plt.grid()
plt.show()

# Plot daily cases worldwide
plt.title('Newly reported cases worldwide (Daily)')
plt.ylabel("Reported cases")
plt.xlabel("Days since start date")
plt.scatter(index, new_cases)
plt.grid()
plt.show()


# Linear regression fit
index = np.asarray(index)
index = index.reshape(-1,1)
X = index

# Linear regression worldwide
Y_world_all = filtered_chart.iloc[:, 4].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor_world = LinearRegression()  # create object for the class
linear_regressor_world.fit(X, Y_world_all)  # perform linear regression
Y_pred = linear_regressor_world.predict(X)  # make predictions
plt.scatter(X, Y_world_all)
plt.title("Total number of cases worldwide \n Linear Regression")
plt.ylabel("Total number of cases")
plt.xlabel("Days since start date")
plt.plot(X, Y_pred, color='red')
plt.grid()
plt.show()

# Linear regression daily cases
Y_world_daily = filtered_chart.iloc[:, 5].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y_world_daily)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
#print(Y_pred)
plt.scatter(X, Y_world_daily)
plt.title("Newly reported cases worldwide (Daily)\n Linear Regression")
plt.ylabel("Reported cases")
plt.xlabel("Days since start date")
plt.plot(X, Y_pred, color='red')
plt.grid()
plt.show()





#################################################
# Section 2. Obtaining linear regression line for total 
#            number of cases in the United States
#################################################

# Column Identifier
iso_code = 'USA'


file_data['date']=pd.to_datetime(file_data['date'])
file_data.date=pd.to_datetime(file_data.date)


# Filters out dates, keeps only the dates of interest (not used)
start_date_time = pd.to_datetime(start_date)
end_date_time = pd.to_datetime(end_date)
filtered_chart = file_data.loc[(file_data['date'] > start_date_time) & (file_data['date'] < end_date_time)]
#print(filtered_chart.dtypes)



# filters to include only U.S. cases
filtered_chart = file_data.loc[(file_data['iso_code'] == iso_code) & (file_data['date'] > start_date) & (file_data['date'] < end_date)]
#print(filtered_chart)
#print("The number of cases around in the U.S. for \nthe dates " + str(start_date) + " through " +  str(end_date))
#print(filtered_chart[columns])



# Store the columns into an array
total_local_cases = np.array(filtered_chart['total_cases'])
new_local_cases = np.array(filtered_chart['new_cases'])
date_timeline = np.array(filtered_chart['date']) # Not used

# count number of days in time period
count = 0
index = []
for i in date_timeline: 
    count += 1

# Make index array
for i in range(0, count): 
    index.append(i)
    
    
plt.title('Total cases in U.S')
plt.ylabel("Total number of cases")
plt.xlabel("Days since start date")
plt.scatter(index, total_local_cases)
plt.grid()
plt.show()


plt.title('Newly reported cases in U.S (Daily)')
plt.ylabel("Reported cases")
plt.xlabel("Days since start date")
plt.scatter(index, new_local_cases)
plt.grid()
plt.show()


# Linear regression  array.reshape(1, -1)
#X = filtered_chart.iloc[:, 0].values.reshape(-1, 1)
index = np.asarray(index)
index = index.reshape(-1,1)
X = index

# Linear regression for all cases USA
Y_US_all = filtered_chart.iloc[:, 4].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y_US_all)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
plt.scatter(X, Y_US_all)
plt.title("Total number of cases in the U.S. \n Linear Regression")
plt.ylabel("Total number of cases")
plt.xlabel("Days since start date")
plt.plot(X, Y_pred, color='red')
plt.grid()
plt.show()

# Linear regression for daily cases USA
Y_US_daily = filtered_chart.iloc[:, 5].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y_US_daily)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
#print(Y_pred)
plt.scatter(X, Y_US_daily)
plt.title("Newly reported cases in the U.S (daily)\n Linear Regression")
plt.ylabel("Reported cases")
plt.xlabel("Days since start date")
plt.plot(X, Y_pred, color='red')
plt.grid()
plt.show()




#################################################
# Section 3. Predicting the number of covid cases 
#            worldwide and in the U.S. 
#################################################

# Days for target end date
days_after = 184

print("By the end of the target date (" + str(target_end_date) + "), the predicted number of cases are as follows:\n")

model_world = LinearRegression()
model_world.fit(X,Y_world_all)

# Put in the day after start date that you would want to predict
X_predict_world = [days_after]  
X_predict_world = np.asarray(X_predict_world)
X_predict_world = X_predict_world.reshape(-1,1)
y_predict_world_all = model_world.predict(X_predict_world)

prediction_value_world_all = y_predict_world_all.item()
prediction_value_world_all = round(prediction_value_world_all)

print("1. Number of covid cases worldwide: " + str(prediction_value_world_all) + "\n")


model_world = LinearRegression()
model_world.fit(X,Y_world_daily)

# Put in the day after start date that you would want to predict
X_predict_world = [days_after]  
X_predict_world = np.asarray(X_predict_world)
X_predict_world = X_predict_world.reshape(-1,1)
y_predict_world_daily = model_world.predict(X_predict_world)

prediction_value_world_daily = y_predict_world_daily.item()
prediction_value_world_daily = round(prediction_value_world_daily)

print("2. Number of newly reported covid cases worldwide per day: " + str(prediction_value_world_daily) + "\n")

model = LinearRegression()
model.fit(X,Y_US_all)


X_predict = [days_after]  
X_predict = np.asarray(X_predict)
X_predict = X_predict.reshape(-1,1)
y_predict_US_all = model.predict(X_predict)

prediction_value_US_all = y_predict_US_all.item()
prediction_value_US_all = round(prediction_value_US_all)

print("3. Number of covid cases in the U.S.: " + str(prediction_value_US_all) + "\n")


model = LinearRegression()
model.fit(X,Y_US_daily)


X_predict = [days_after]  
X_predict = np.asarray(X_predict)
X_predict = X_predict.reshape(-1,1)
y_predict_US_daily = model.predict(X_predict)

prediction_value_US_daily = y_predict_US_daily.item()
prediction_value_US_daily = round(prediction_value_US_daily)

print("4. Number of newly reported covid cases in the U.S per day: " + str(prediction_value_US_daily) + "\n")




#################################################
# Section 4. obtaing actual results 
#################################################

Y_US_all = filtered_chart.iloc[:, 4].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
print("Actual results according to data: ")

iso_code = 'OWID_WRL'
actual_result_world_all = file_data.loc[(file_data['iso_code'] == iso_code) & (file_data['date'] == target_end_date)]
actual_result_world_all = round(actual_result_world_all['total_cases'].values.reshape(-1, 1).item())
print("1. " + str(actual_result_world_all))

actual_result_world_daily = file_data.loc[(file_data['iso_code'] == iso_code) & (file_data['date'] == target_end_date)]
actual_result_world_daily = round(actual_result_world_daily['new_cases'].values.reshape(-1, 1).item())
print("2. " + str(actual_result_world_daily))


iso_code = 'USA'
actual_result_US_all = file_data.loc[(file_data['iso_code'] == iso_code) & (file_data['date'] == target_end_date)]
actual_result_US_all = round(actual_result_US_all['total_cases'].values.reshape(-1, 1).item())
print("3. " + str(actual_result_US_all))

actual_result_US_daily = file_data.loc[(file_data['iso_code'] == iso_code) & (file_data['date'] == target_end_date)]
actual_result_US_daily = (round(actual_result_US_daily['new_cases'].values.reshape(-1, 1).item()))
print("4. " + str(actual_result_US_daily)+ "\n")



#################################################
# Section 5. Calculating percent error 
#################################################

print("Calculating percent error for each case")

# Percent error formula
def percent_error(theoretical, actual): 
    error = abs(((theoretical - actual)/actual) * 100)
    return error
    

case_1 = percent_error(prediction_value_world_all, actual_result_world_all)
case_2 = percent_error(prediction_value_world_daily, actual_result_world_daily)
case_3 = percent_error(prediction_value_US_all, actual_result_US_all)
case_4 = percent_error(prediction_value_US_daily, actual_result_US_daily)

print("1. " + str(case_1) + "% error")
print("2. " + str(case_2) + "% error")
print("3. " + str(case_3) + "% error")
print("4. " + str(case_4) + "% error")
print("\nPlease read attached PDF file for possible explanations for such high errors")


#################################################
# Section 6. Plotting up to date results just 
#            for fun
#################################################

first_start_date = '2020-01-21'
last_end_date = '2020-10-21'

iso_code = 'OWID_WRL'
all_time_chart = file_data.loc[(file_data['iso_code'] == iso_code) & (file_data['date'] >= first_start_date) & (file_data['date'] < last_end_date )]

                                                                                                       
# Store the columns into an array
all_time_total_cases = np.array(all_time_chart['total_cases'])
all_time_new_cases = np.array(all_time_chart['new_cases'])

date_timeline = np.array(all_time_chart['date']) 

count = 0
index = []
for i in date_timeline: 
    count += 1

for i in range(0, count): 
    index.append(i)


# Plot cases worldwide
plt.title('All-time total cases worldwide')
plt.ylabel("Total number of cases")
plt.xlabel("Days since start date")
plt.scatter(index, all_time_total_cases)
plt.grid()
plt.show()


# Plot daily cases worldwide
plt.title('Newly reported cases worldwide (Daily)')
plt.ylabel("Reported cases")
plt.xlabel("Days since start date")
plt.scatter(index, all_time_new_cases)
plt.grid()
plt.show()



iso_code = 'USA'
all_time_chart = file_data.loc[(file_data['iso_code'] == iso_code) & (file_data['date'] >= first_start_date) & (file_data['date'] < last_end_date)]

# Store the columns into an array
all_time_total_cases = np.array(all_time_chart['total_cases'])
all_time_new_cases = np.array(all_time_chart['new_cases'])

date_timeline = np.array(all_time_chart['date']) 

count = 0
index = []
for i in date_timeline: 
    count += 1

for i in range(0, count): 
    index.append(i)

# Plot cases worldwide
plt.title('All-time total cases in the U.S.')
plt.ylabel("Total number of cases")
plt.xlabel("Days since start date")
plt.plot(index, all_time_total_cases)
plt.grid()
plt.show()


# Plot daily cases worldwide
plt.title('Newly reported cases in the U.S (Daily)')
plt.ylabel("Reported cases")
plt.xlabel("Days since start date")
plt.plot(index, all_time_new_cases)
plt.grid()
plt.show()











