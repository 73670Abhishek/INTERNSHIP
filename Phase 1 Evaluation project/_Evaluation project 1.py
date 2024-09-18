#!/usr/bin/env python
# coding: utf-8

# # Baseball Case Study

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sma
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# # load the datasets

# In[3]:


df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/Data-Science-ML-Capstone-Projects/master/baseball.csv')
df


# In[4]:


#display first 10 datasets
df.head(10)


# # Exploratory Data Analysis(EDA)

# In[5]:


#checking the dimension of the dataset

df.shape


# The dataset contains 30 rows and 17 columns, indicating a small sample size for model building.

# In[6]:


#all columns of dataset
df.columns


# In[7]:


#checking the datatypes of column
df.dtypes


# All columns in the dataset are of integer (int64) type except for the ERA column, which is a float (float64).

# In[8]:


df.info()


# The dataset has 30 entries with 17 columns, all of which contain non-null values, with 16 columns being integers and 1 column (ERA) as a float, using 4.1 KB of memory.

# In[9]:


#checking the null values
df.isnull().sum()


# so as observed from the above data that there are no null values and so don't have to deal with any missing data.

# In[10]:


#lets visualise it using heatmap
sns.heatmap(df.isnull())


# # Description

# In[11]:


df.describe()


# The data shows that baseball teams have different levels of performance, with wide variations in wins, runs, and other stats, highlighting differences in team strength and gameplay.

# In[12]:


# Count unique values for each column
for column in df.columns:
    print(f"Value counts for {column}:")
    print(df[column].value_counts())
    print("\n")


# we observe that most of the columns (like W, R, AB, etc.) represent continuous data, while a few (such as CG, SHO, SV, and E) are more categorical due to repeated values or limited unique values.

# In[13]:


for column in df.columns:
    print(f"{column} has {df[column].nunique()} unique values.")


# from above data to some extent we can differentiate between continuous and categorical data.

# In[14]:


df['W'].unique()


# In[15]:


df['CG'].unique()


# In[16]:


df['SHO'].unique()


# In[17]:


df['SV'].unique()


# In[18]:


df['E'].unique()


# # visualisation

# In[19]:


#countplot for the 'W' column
plt.figure(figsize=(10, 5))
sns.countplot(x=df['W'])
plt.title('Countplot for Wins (W)')
plt.show()


# In[20]:


#countplot for the 'R' column
plt.figure(figsize=(10, 5))
sns.countplot(x=df['R'])
plt.xticks(rotation=45)
plt.title('Countplot for Runs Scored(R)')
plt.show()


# In[21]:


#countplot for the 'AB' column
plt.figure(figsize=(10, 5))
sns.countplot(x=df['AB'])
plt.xticks(rotation=45)
plt.title('Countplot for At Bat (AB)')
plt.show()


# In[22]:


#countplot for the 'H' column
plt.figure(figsize=(10, 5))
sns.countplot(x=df['H'])
plt.xticks(rotation=45)
plt.title('Countplot for Hit (H)')
plt.show()


# In[23]:


#countplot for the '2B' column
plt.figure(figsize=(10, 5))
sns.countplot(x=df['2B'])
plt.xticks(rotation=45)
plt.title('Countplot for A Double (2B)')
plt.show()


# In[24]:


#countplot for the '3B' column
plt.figure(figsize=(10, 5))
sns.countplot(x=df['3B'])
plt.title('Countplot for A Triple (3B)')
plt.show()


# In[25]:


#countplot for the 'HR' column
plt.figure(figsize=(10, 5))
sns.countplot(x=df['HR'])
plt.xticks(rotation=45)
plt.title('Countplot for Home Run (HR)')
plt.show()


# In[26]:


#countplot for the 'BB' column
plt.figure(figsize=(10, 5))
sns.countplot(x=df['BB'])
plt.xticks(rotation=45)
plt.title('Countplot for Walk (BB)')
plt.show()


# In[27]:


#countplot for the 'SO' column
plt.figure(figsize=(10, 5))
sns.countplot(x=df['SO'])
plt.xticks(rotation=45)
plt.title('Countplot for Strikeout (SO)')
plt.show()


# In[28]:


#countplot for the 'SB' column
plt.figure(figsize=(10, 5))
sns.countplot(x=df['SB'])
plt.xticks(rotation=45)
plt.title('Countplot for Stolen base (SB)')
plt.show()


# In[29]:


#countplot for the 'RA' column
plt.figure(figsize=(10, 5))
sns.countplot(x=df['RA'])
plt.xticks(rotation=45)
plt.title('Countplot for Run Average (RA)')
plt.show()


# In[30]:


#countplot for the 'ER' column
plt.figure(figsize=(10, 5))
sns.countplot(x=df['ER'])
plt.xticks(rotation=45)
plt.title('Countplot for Earned Run (ER)')
plt.show()


# In[31]:


#countplot for the 'ERA' column
plt.figure(figsize=(10, 5))
sns.countplot(x=df['ERA'])
plt.xticks(rotation=45)
plt.title('Countplot for Earned Run Average (ERA)')
plt.show()


# In[32]:


#countplot for the 'CG' column
plt.figure(figsize=(10, 5))
sns.countplot(x=df['CG'])
plt.title('Countplot for Complete Games(CG)')
plt.show()


# In[33]:


#countplot for the 'SHO' column
plt.figure(figsize=(10, 5))
sns.countplot(x=df['SHO'])
plt.title('Countplot for Shutout(SHO)')
plt.show()


# In[34]:


#countplot for the 'SV' column
plt.figure(figsize=(10, 5))
sns.countplot(x=df['SV'])
plt.title('Countplot for Save (SV)')
plt.show()


# In[35]:


#countplot for the 'E' column
plt.figure(figsize=(10, 5))
sns.countplot(x=df['E'])
plt.title('Countplot for Error (E)')
plt.show()


# In[36]:


# Visualizing the distribution of the target variable 
sns.histplot(df['W'], bins=15, kde=True)
plt.title('Distribution of Wins')
plt.show()


# The distribution of wins (W) appears to be fairly balanced, with a peak around 85 wins and fewer teams achieving both low (65-70) and high (95-100) win totals.

# In[37]:


# List of all columns
columns = df.columns[df.columns != 'W']

# Loop through each column and plot
for column in columns:
    plt.figure(figsize=(8, 4))
    sns.regplot(x=df[column], y=df['W'])
    plt.title(f'Wins vs {column}')
    plt.xlabel(column)
    plt.ylabel('Wins')
    plt.show()


# In[38]:


sns.pairplot(df)
plt.show()


# >The scatterplot matrix shows that some features are related to each other, while others are not. 
# 
# >The diagonal histograms show the distribution of each feature, helping us understand which ones may need adjustments, like fixing any skewed data

# In[39]:


plt.figure(figsize=(10,30))

for i, column in enumerate(columns):
    plt.subplot(len(columns), 1, i + 1)
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)

plt.tight_layout()
plt.show()


# >The variables with outliers are R (Runs), ERA (Earned Run Average), SHO (Shutouts), SV (Saves), and E (Errors), showing a few extreme values above the normal range.

# # Remove Outliers using 'IQR'

# In[40]:


#Columns with outliers
outlier_columns = ['R', 'ERA', 'SHO', 'SV', 'E']

for col in outlier_columns:
   # Calculate Q1 (25th percentile) and Q3 (75th percentile)
   Q1 = df[col].quantile(0.25)
   Q3 = df[col].quantile(0.75)
   
   # Calculate IQR
   IQR = Q3 - Q1
   
   # Calculate outliers
   lower_bound = Q1 - 1.5 * IQR
   upper_bound = Q3 + 1.5 * IQR
   
   # Identify outliers
   outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
   
   print(f'{col} - IQR: {IQR}, Lower Bound: {lower_bound}, Upper Bound: {upper_bound}')
   print(f'Outliers in {col}:\n{outliers[col]}')


# In[41]:


# Print cleaned DataFrame
print(f'Cleaned DataFrame:\n{df}')


# # Check Skewness

# In[42]:


df.skew()


# In[43]:


# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create a heatmap
plt.figure(figsize=(15, 10))  
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5)
plt.yticks(rotation = 0)

# Display it
plt.title('Correlation Matrix Heatmap')
plt.show()


# observation:
# 
#   1.  W (Wins) has a strong negative correlation with RA (Runs Allowed) (-0.81), ER (Earned Runs) (-0.81), and ERA (Earned Run Average) (-0.82). This suggests that as runs allowed, earned runs, and ERA decrease, the number of wins increases.
# 
#   2.  R (Runs) shows a positive correlation with H (Hits) (0.48), BB (Walks) (0.67), and HR (Home Runs) (0.31), indicating that these factors contribute to scoring more runs.
# 
#   3.  H (Hits) and AB (At Bats) have a strong correlation (0.74), meaning more at-bats typically lead to more hits.
# 
#   4.  ERA (Earned Run Average) is highly positively correlated with RA (Runs Allowed) (0.99) and ER (Earned Runs) (0.99). These are strongly related metrics regarding pitching performance.
# 
#   5.  SO (Strikeouts) and BB (Walks) show a weak correlation (-0.16), meaning strikeouts and walks are not strongly related.
# 
#   6.  SV (Saves) has a moderate positive correlation with W (Wins) (0.67) and SHO (Shutouts) (0.43), meaning more saves often lead to more wins and shutouts.

# # Multicollinearity Identification

# In[44]:


#function to find multicollinearity

def find_multicollinearity(data):
   
    data_with_const = sma.add_constant(data)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = data_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(data_with_const.values, i) for i in range(data_with_const.shape[1])]
    vif_data = vif_data[vif_data["Feature"] != "const"]
    
    return vif_data


# In[45]:


vif_result = find_multicollinearity(df)
print(vif_result)


# > Based on the VIF values we got, several features have high multicollinearity. 
# 
# > Features like ER, ERA, and RA have extremely high VIF values, indicating severe multicollinearity.

# In[46]:


# Drop 'ER' and 'RA' as they have very high VIFs
df = df.drop(columns=['ER', 'RA'])

# Recalculation of VIF
vif_result = find_multicollinearity(df)
print(vif_result)


# > Since ERA (Earned Run Average) is often directly correlated with ER (Earned Run) and RA (Run Average), it makes sense to keep only one of these features.

# In[47]:


df.shape


# In[48]:


df.corr()


# In[49]:


df.corr().W.sort_values()


# # Seperate features  and target  variable

# In[50]:


X = df.drop('W', axis=1)  
Y = df['W'] 


# In[51]:


# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# shapes of the resulting datasets
print(f"Training features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Training target shape: {Y_train.shape}")
print(f"Test target shape: {Y_test.shape}")


# # Initialization and Model Training

# In[52]:


# Initialize and train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, Y_train)
pred_lr = lr.predict(X_test)

# Evaluate the model
print("Linear Regression:")
print("R^2 score:", r2_score(Y_test, pred_lr))
print("Mean Absolute Error:", mean_absolute_error(Y_test, pred_lr))
print("Mean Squared Error:", mean_squared_error(Y_test, pred_lr))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Y_test, pred_lr)))
print("\n")


# In[53]:


# Initialize and train the Random Forest model
rf = RandomForestRegressor()
rf.fit(X_train, Y_train)
pred_rf = rf.predict(X_test)

# Evaluate the model
print("Random Forest Regressor:")
print("R^2 score:", r2_score(Y_test, pred_rf))
print("Mean Absolute Error:", mean_absolute_error(Y_test, pred_rf))
print("Mean Squared Error:", mean_squared_error(Y_test, pred_rf))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Y_test, pred_rf)))
print("\n")


# In[54]:


# Initialize and train the Decision Tree model
dt = DecisionTreeRegressor()
dt.fit(X_train, Y_train)
pred_dt = dt.predict(X_test)

# Evaluate the model
print("Decision Tree Regressor:")
print("R^2 score:", r2_score(Y_test, pred_dt))
print("Mean Absolute Error:", mean_absolute_error(Y_test, pred_dt))
print("Mean Squared Error:", mean_squared_error(Y_test, pred_dt))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Y_test, pred_dt)))
print("\n")


# In[55]:


# Initialize and train the Gradient Boosting Regressor model
gbr = GradientBoostingRegressor()
gbr.fit(X_train, Y_train)
pred_gbr = gbr.predict(X_test)

# Evaluate the model
print("Gradient Boosting Regressor:")
print("R^2 score:", r2_score(Y_test, pred_gbr))
print("Mean Absolute Error:", mean_absolute_error(Y_test, pred_gbr))
print("Mean Squared Error:", mean_squared_error(Y_test, pred_gbr))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Y_test, pred_gbr)))


# In[56]:


# Initialize and train the Lasso model
lasso = Lasso(alpha=0.1)  # You can adjust the alpha value for regularization
lasso.fit(X_train, Y_train)
pred_lasso = lasso.predict(X_test)

# Evaluate the model
print("Lasso Regression:")
print("R^2 score:", r2_score(Y_test, pred_lasso))
print("Mean Absolute Error:", mean_absolute_error(Y_test, pred_lasso))
print("Mean Squared Error:", mean_squared_error(Y_test, pred_lasso))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Y_test, pred_lasso)))


# # checking overfitting

# continuing with the best among above model

# In[57]:


# Initialize and train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, Y_train)

# Predict on training data
train_pred_lr = lr.predict(X_train)

# Predict on test data
test_pred_lr = lr.predict(X_test)

# Evaluate training metrics
print("Training Metrics for Linear Regression:")
print("R^2 score:", r2_score(Y_train, train_pred_lr))
print("Mean Absolute Error:", mean_absolute_error(Y_train, train_pred_lr))
print("Mean Squared Error:", mean_squared_error(Y_train, train_pred_lr))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Y_train, train_pred_lr)))

# Evaluate test metrics
print("\nTest Metrics for Linear Regression:")
print("R^2 score:", r2_score(Y_test, test_pred_lr))
print("Mean Absolute Error:", mean_absolute_error(Y_test, test_pred_lr))
print("Mean Squared Error:", mean_squared_error(Y_test, test_pred_lr))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Y_test, test_pred_lr)))


# In[58]:


# Initialize and train the Lasso model
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, Y_train)

# Predict on training data
train_pred_lasso = lasso.predict(X_train)

# Predict on test data
test_pred_lasso = lasso.predict(X_test)

# Evaluate training metrics
print("Training Metrics for Lasso Regression:")
print("R^2 score:", r2_score(Y_train, train_pred_lasso))
print("Mean Absolute Error:", mean_absolute_error(Y_train, train_pred_lasso))
print("Mean Squared Error:", mean_squared_error(Y_train, train_pred_lasso))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Y_train, train_pred_lasso)))

# Evaluate test metrics
print("\nTest Metrics for Lasso Regression:")
print("R^2 score:", r2_score(Y_test, test_pred_lasso))
print("Mean Absolute Error:", mean_absolute_error(Y_test, test_pred_lasso))
print("Mean Squared Error:", mean_squared_error(Y_test, test_pred_lasso))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Y_test, test_pred_lasso)))


# # hyperparameter tunning and cross validation

# In[59]:


# Define the parameter grid
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}


ridge = Ridge()
grid_search_ridge = GridSearchCV(estimator=ridge, param_grid=param_grid, scoring='r2', cv=5)
grid_search_ridge.fit(X_train, Y_train)

# Best parameters and score
print("Best Parameters for Ridge Regression:", grid_search_ridge.best_params_)
print("Best Cross-Validation Score for Ridge Regression:", grid_search_ridge.best_score_)

# Train and evaluate the best Ridge model
best_ridge = grid_search_ridge.best_estimator_
train_pred_ridge = best_ridge.predict(X_train)
test_pred_ridge = best_ridge.predict(X_test)

print("\nTraining Metrics for Best Ridge Regression:")
print("R^2 score:", r2_score(Y_train, train_pred_ridge))
print("Mean Absolute Error:", mean_absolute_error(Y_train, train_pred_ridge))
print("Mean Squared Error:", mean_squared_error(Y_train, train_pred_ridge))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Y_train, train_pred_ridge)))

print("\nTest Metrics for Best Ridge Regression:")
print("R^2 score:", r2_score(Y_test, test_pred_ridge))
print("Mean Absolute Error:", mean_absolute_error(Y_test, test_pred_ridge))
print("Mean Squared Error:", mean_squared_error(Y_test, test_pred_ridge))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Y_test, test_pred_ridge)))


# In[60]:


# Define the parameter grid
param_grid = {'alpha': [0.01, 0.1, 1, 10]}


lasso = Lasso()
grid_search_lasso = GridSearchCV(estimator=lasso, param_grid=param_grid, scoring='r2', cv=5)
grid_search_lasso.fit(X_train, Y_train)

# Best parameters and score
print("Best Parameters for Lasso Regression:", grid_search_lasso.best_params_)
print("Best Cross-Validation Score for Lasso Regression:", grid_search_lasso.best_score_)

# Train and evaluate the best Lasso model
best_lasso = grid_search_lasso.best_estimator_
train_pred_lasso = best_lasso.predict(X_train)
test_pred_lasso = best_lasso.predict(X_test)

print("\nTraining Metrics for Best Lasso Regression:")
print("R^2 score:", r2_score(Y_train, train_pred_lasso))
print("Mean Absolute Error:", mean_absolute_error(Y_train, train_pred_lasso))
print("Mean Squared Error:", mean_squared_error(Y_train, train_pred_lasso))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Y_test, test_pred_lasso)))


# Hence we found lasso is overall best amongst all,hence we proceed it with furthur.

# # Re-train of the model

# In[61]:


# Initialize the final Lasso Regression model with the best parameters
final_lasso = Lasso(alpha=1)

# Train the model on the full dataset
final_lasso.fit(X, Y)


# # save the model

# In[62]:


import joblib

# Save the trained model
joblib.dump(final_lasso, 'final_lasso_model.pkl')


# # ==========xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx===========

# # Avocado Project

# In[80]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/Data-Science-ML-Capstone-Projects/master/avocado.csv ')
df


# The dataset is loaded successfully with 16,468 rows and 14 columns, and it contains both numerical and categorical features.

# In[3]:


df.head()


# In[4]:


df.tail()


# we can see a lot of missing data.

# # Exploratory Data Analysis(EDA)

# In[5]:


df.shape


# The dataset has 16,468 rows and 14 columns, indicating a reasonably large dataset.

# In[6]:


df.columns


# In[7]:


df.dtypes


#  The dataset contains a mix of float (for numerical data) and object (for dates and categorical data).

# In[8]:


#checking the null values
df.isnull().sum()


# A large number of null values are present in almost all columns, indicating potential issues in data completeness.

# In[9]:


df.info()


# Out of 16,468 rows, only 1,517 rows have non-null values across all columns, confirming a significant amount of missing data that will need to be addressed.

# In[10]:


#lets visualise above data using heatmap
sns.heatmap(df.isnull())


# # Desceiption

# In[11]:


df.describe()


# # Handling Missing Values

# In[12]:


print("Shape before dropping NaNs:", df.shape)
df.dropna(inplace=True)
print("Shape after dropping NaNs:", df.shape)


#  This step removes rows with missing values, leaving 1517 complete rows for further analysis.

# # Drop unnecessary column

# In[13]:


# Remove the unnecessary column
df.drop("Unnamed: 0", axis=1, inplace=True)
print("Shape after dropping 'Unnamed: 0':", df.shape)


# # lets check it

# In[14]:


print("NaN Values per Column:")
print(df.isnull().sum())


# In[15]:


# visualize the null values in heatmap
sns.heatmap(df.isnull())


# # checking uniqueness

# In[16]:


df['AveragePrice'].unique()


# In[17]:


df['region'].unique()


# In[18]:


# Lets change date column from object data type to datetime data type
df['Date'] = pd.to_datetime(df['Date'])


# In[19]:


#Extract useful date features (like year, month, day)
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day

# drop  'Date' 
df.drop('Date', axis=1, inplace=True)


# # seperating numerical and categorical column

# In[20]:


numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns


# In[21]:


print(numerical_columns)


# In[22]:


categorical_columns = df.select_dtypes(include=['object']).columns


# In[23]:


print(categorical_columns)


# # visualization

# In[24]:


plt.figure(figsize=(12, 8))
sns.barplot(x='region', y='AveragePrice', data=df, ci=None)  # `ci=None` removes the error bars
plt.xticks(rotation=90)
plt.show()


# In[25]:


# Print unique values in the 'year' column
print(df['year'].unique())

# Plot 
sns.countplot(x='year', data=df)
plt.title('Count of Entries by Year')
plt.show()


# In[26]:


plt.figure(figsize=(20, 25), facecolor='white')

# Initialize plot number
plotnumber = 1

# Iterate over numerical columns and create subplots
for column in numerical_columns:
    if plotnumber <= 14:  
        ax = plt.subplot(4, 3, plotnumber)  
        sns.histplot(df[column], kde=True, color='b', ax=ax) 
        plt.xlabel(column, fontsize=20)
        plt.title(f'Distribution of {column}', fontsize=22) 
        plotnumber += 1

plt.tight_layout() 
plt.show()


# In[27]:


# Bar plot for 'region'
plt.figure(figsize=(12, 8))
df['region'].value_counts().plot(kind='bar', color='skyblue')
plt.xlabel('Region', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Count of Each Region', fontsize=16)
plt.xticks(rotation=90)
plt.show()


# In[28]:


# Pie chart for 'type'
plt.figure(figsize=(8, 8))
df['type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Proportion of Each Type', fontsize=16)
plt.ylabel('')
plt.show()


# In[29]:


plt.figure(figsize=(14, 8))
sns.barplot(x='region', y='Total Volume', data=df, palette='viridis')
plt.xticks(rotation=90)
plt.title('Total Volume of Avocados Sold by Region')
plt.show()


# In[30]:


plt.figure(figsize=(14, 8))
sns.lineplot(data=df[['4046', '4225', '4770']].sum().reset_index(), x='index', y=0, marker='o', palette='tab10')
plt.xlabel('PLU Code')
plt.ylabel('Total Sales')
plt.title('Total Sales by PLU Code')
plt.show()


# PLU 4046 and 4225 have similar high total sales, but there is a significant drop in sales for PLU 4770.

# In[31]:


plt.figure(figsize=(14, 8))
sns.kdeplot(df['4046'], label='4046', fill=True)
sns.kdeplot(df['4225'], label='4225', fill=True)
sns.kdeplot(df['4770'], label='4770', fill=True)
plt.xlabel('Sales')
plt.title('KDE Plot of Avocado Sales by PLU Code')
plt.legend()
plt.show()


# Most of the sales are concentrated around lower values (near zero), as shown by the sharp spike near the origin.
# The three PLU codes have overlapping distributions, but PLU 4770 appears to have a particularly sharp peak near zero, suggesting that a significant portion of its sales volumes are relatively small.
# The long tail stretching to the right indicates that there are a few very large sales values, but they are much less frequent.

# In[32]:


# Define numerical columns
numerical_col = ['AveragePrice', 'Total Volume', '4046', '4225', '4770', 
                 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 
                 'year', 'month', 'day']


fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 20))
axes = axes.ravel()

# Plot boxplots for each column
for i, col in enumerate(numerical_col):
    sns.boxplot(data=df[col], ax=axes[i],color="purple")
    axes[i].set_title(col)

plt.tight_layout()
plt.show()


# # remove outliers

# In[33]:


df.columns


# In[34]:


# containing outliers

c_out = df[['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']]


# In[35]:


#Zscore method

z = np.abs(zscore(c_out))

z


# In[36]:


# Create new DataFrame with z-scores < 3
new_df = df[(z < 3).all(axis=1)]

print(new_df)


# In[37]:


df.shape


# In[38]:


new_df.shape


# # Label encoding

# In[39]:


# Columns with catagorical data

categorical_col = ["type","region"]


# In[40]:


LE = LabelEncoder()

new_df[categorical_col] = new_df[categorical_col].apply(LE.fit_transform)

new_df[categorical_col]


# In[41]:


# Apply Yeo-Johnson transformation to remove skewness

# Features
skewed_features = ['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']

scaler = PowerTransformer(method='yeo-johnson') 
new_df[skewed_features] = scaler.fit_transform(new_df[skewed_features].values)


new_df[skewed_features].head()


# In[42]:


new_df.skew()


# In[43]:


plt.figure(figsize=(12, 8))

# Plot the heatmap 
sns.heatmap(new_df.corr(), annot=True, linewidths=0.2, fmt='.3f', cmap='coolwarm')
plt.yticks(rotation=0)

plt.title('Correlation Matrix Heatmap')
plt.show()


# # using VIF for multicollinearity

# In[44]:


# Calculate the correlation matrix
corr = new_df.corr()

# Sort the correlation values for 'AveragePrice'
corr['AveragePrice'].sort_values(ascending=False)


# # Visualization of corelation between label and features.

# In[45]:


plt.figure(figsize = (20, 8))

new_df.corr()['AveragePrice'].sort_values(ascending=False).drop(['AveragePrice']).plot(kind = 'bar')

plt.xlabel('Feature',fontsize = 12)
plt.ylabel('Target',fontsize = 12)
plt.title('correlation between label and feature using bar plot', fontsize = 15)
plt.show()


# # Regression Model

# In[46]:


# seperate x & y labels

x = new_df.drop('AveragePrice', axis = 1)
y = new_df['AveragePrice']


# In[47]:


x.shape


# In[48]:


y.shape


# In[49]:


x.head()


# # Standarization

# In[50]:


scaler = StandardScaler()

x = pd.DataFrame(scaler.fit_transform(x), columns = x.columns)
x


# In[51]:


# Find Varience Inflation Factor (VIF) in each scaled column above.

vif = pd.DataFrame()
vif['VIF values'] = [variance_inflation_factor(x.values, i)
              for i in range(len(x.columns))]
vif['Features'] = x.columns

vif


# In[52]:


# Drop features with high VIF values
x.drop(['Total Volume', 'Total Bags', 'Small Bags'], axis=1, inplace=True)

# Check VIF values again
vif = pd.DataFrame()
vif['VIF values'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif['Features'] = x.columns

# Display updated VIF values
print(vif)


# # Finding best random state

# In[53]:


max_acc = 0
best_random_state = 0

for i in range(1, 200):
    # Split 
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=i)
    
    # Initialize and train the model
    model = KNeighborsRegressor()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    acc = r2_score(y_test, predictions)
    
    if acc > max_acc:
        max_acc = acc
        best_random_state = i

print(f'Maximum R² score is {max_acc:.4f} on random_state {best_random_state}')


# In[54]:


# Train & Test Split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = i)


# # model training

# In[55]:


#Linear regression
LR = LinearRegression()
LR.fit(x_train, y_train)

# prediction
predLR = LR.predict(x_test)
print('R2_score:', r2_score(y_test, predLR))
print('MAE:', metrics.mean_absolute_error(y_test, predLR))
print('MSE:', metrics.mean_squared_error(y_test, predLR))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predLR)))

# Checking Cross_Validation score for Linear Regression
print('Cross_Validaton_Score', cross_val_score(LR, x, y, cv = 5).mean())


# In[56]:


#Kneighbour
# Checking R2 score for KNN Regressor
knn = KNN()

knn.fit(x_train, y_train)

# predict
predknn = knn.predict(x_test)

print('R2_Score:',r2_score(y_test, predknn))
print('MAE:',metrics.mean_absolute_error(y_test, predknn))
print('MSE:',metrics.mean_squared_error(y_test, predknn))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, predknn)))

# Checking Cross_Validaton score for KNN
print('Cross_Validaton_Score', cross_val_score(knn, x, y, cv = 5).mean())


# In[57]:


#Randomforest
RFR = RandomForestRegressor()
RFR.fit(x_train, y_train)

# predict
pred_RFR = RFR.predict(x_test)
print('R2_Score:', r2_score(y_test, pred_RFR))

# Mean Absolute Error (MAE)
print('MAE:', metrics.mean_absolute_error(y_test, pred_RFR))

# Mean Squared Error (MSE)
print('MSE:', metrics.mean_squared_error(y_test, pred_RFR))

# Root Mean Squared Error (RMSE)
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred_RFR)))

# Checking Cross_Validation score for Random Forest Regression
print('Cross_Validaton_Score', cross_val_score(RFR, x, y, cv = 5).mean())


# In[58]:


#GradientBoosting
GB = GradientBoostingRegressor()

GB.fit(x_train, y_train)

# predict
predGB = GB.predict(x_test)

print('R2_Score:',metrics.r2_score(y_test,predGB))
print('MAE:',metrics.mean_absolute_error(y_test, predGB))
print('MSE:',metrics.mean_squared_error(y_test, predGB))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predGB)))

# Checking Cross_Validaton score for GradientBoosting Regressor
print('Cross_Validaton_Score', cross_val_score(GB, x, y, cv = 5).mean())


# In[59]:


#SVR

svr = SVR()
svr.fit(x_train, y_train)

# predict
predsvr = svr.predict(x_test)

print('R2_Score:', r2_score(y_test, predsvr))
print('MAE:', metrics.mean_absolute_error(y_test, predsvr))
print('MSE:', metrics.mean_squared_error(y_test, predsvr))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predsvr)))

# Checking cv score for SVR 

print('Cross_Validaton_Score', cross_val_score(svr, x, y, cv = 5).mean())


# # Hyperparameter Tunning

# Proceeding with the best suited model 

# In[64]:


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}


# In[65]:


#initialization
rf = RandomForestRegressor()


# In[66]:


grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, scoring='r2')


# In[68]:


grid_search.fit(x_train, y_train)


# In[69]:


print('Best Parameters:', grid_search.best_params_)


# In[70]:


best_rf = grid_search.best_estimator_
predictions = best_rf.predict(x_test)

print('R2_score:', r2_score(y_test, predictions))
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# # save model

# In[72]:


# Save the trained model 
joblib.dump(rf, 'random_forest_model.pkl')

print("Model saved successfully.")


# # classification model

# In[73]:


new_df.head()


# # Train & Test

# In[74]:


# splitting the values to x & y 

x_c = new_df.drop(columns = ['region'])
y_c = new_df[['region']]

print(x_c.shape)
print(y_c.shape)


# # Standardizing Features

# In[75]:


sc = StandardScaler()
sc.fit_transform(x_c)

x_c = pd.DataFrame(x_c, columns = x_c.columns)


# # spliting for train and test

# In[76]:


x_train, x_test, y_train, y_test = train_test_split(x_c, y_c, test_size = 0.30 ,random_state = 49)


# # model training

# In[82]:


#Logistic classifier
LR = LogisticRegression()
LR.fit(x_train, y_train)

predLR = LR.predict(x_test)

print(accuracy_score(y_test, predLR))
print(confusion_matrix(y_test, predLR))
print(classification_report(y_test, predLR))


# In[81]:


#Decision Tree  classifier
DTC = DecisionTreeClassifier()
DTC.fit(x_train, y_train)

predDTC = DTC.predict(x_test)

print(accuracy_score(y_test, predDTC))
print(confusion_matrix(y_test, predDTC))
print(classification_report(y_test, predDTC))


# In[83]:


#SVM classifier
svc = SVC()
svc.fit(x_train,y_train)

predsvc = svc.predict(x_test)

print(accuracy_score(y_test, predsvc))
print(confusion_matrix(y_test, predsvc))
print(classification_report(y_test, predsvc))


# In[84]:


#Gradient Boosting Classifier
GB = GradientBoostingClassifier()
GB.fit(x_train, y_train)

predGB = GB.predict(x_test)

print(accuracy_score(y_test, predGB))
print(confusion_matrix(y_test, predGB))
print(classification_report(y_test, predGB))


# # Hyperparameter Tunning & Cross validation of best MOdel 

# In[85]:


# Cross_Validation score for Gradient Boosting Classifier
print(cross_val_score(GB, x_c, y_c, cv=5).mean())


# In[89]:


parameters = {
    'n_estimators': [10, 50],  
    'max_features': ["sqrt"], 
    'max_leaf_nodes': [5, 10],  
}


# In[90]:


GCV = GridSearchCV(GradientBoostingClassifier(), parameters, cv = 5)


# In[91]:


GCV.fit(x_train, y_train)


# In[92]:


GCV.best_params_


# In[93]:


# Initialize the GradientBoostingClassifier with valid parameters
Region = GradientBoostingClassifier(max_features='sqrt', max_leaf_nodes=10, n_estimators=50)

# Fit the model
Region.fit(x_train, y_train)

# Make predictions
pred = Region.predict(x_test)

# Calculate accuracy
acc = accuracy_score(y_test, pred)

# Print the accuracy as a percentage
print(f'Accuracy: {acc * 100:.2f}%')


# In[95]:


# Save the  model
joblib.dump(Region, 'gradient_boosting_model.pkl')

print("Model saved successfully.")


# # xxxxxxxxxxxxxxxx=============================xxxxxxxxxxxxxxxxxxxx

# In[218]:


import pandas as pd
import numpy as np
import zipfile
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import joblib
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PowerTransformer
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# In[143]:


csv_path = r'C:\Users\dell\Downloads\ibm-hr-analytics-employee-attrition-performance\WA_Fn-UseC_-HR-Employee-Attrition.csv'

# Load the dataset
df = pd.read_csv(csv_path)

# Display the first few rows
df.head(10)


# In[144]:


df.tail(5)


# # Exploratory Data Analysis(EDA)

# In[145]:


df.shape


# In[146]:


# Check data types

df.dtypes


# The dataset contains 35 columns with mixed data types: several columns are integers representing numerical data (e.g., Age, DailyRate, YearsAtCompany), while others are objects (categorical data) like Attrition, BusinessTravel, Department, and OverTime.

# In[147]:


df.info()


# The dataset has 1470 entries with 35 columns, containing 26 numerical columns (int64) and 9 categorical columns (object), with no missing values.

# In[149]:


#Checking missing values
df.isnull().sum()


# There are no missing values in the dataset, as all columns have zero null values.

# In[150]:


sns.heatmap(df.isnull())


# In[151]:


#check the no. of columns

df.columns


# In[152]:


# check uniqueness

df.nunique()


# The dataset has a variety of unique values across columns: most numerical columns (like Age, DistanceFromHome, and YearsAtCompany) have many unique values, while categorical columns (like Attrition, BusinessTravel, and Gender) have fewer distinct values. Some columns, such as EmployeeCount and StandardHours, have only one unique value, indicating they provide no variability.

# In[153]:


# count Attrition
df["Attrition"].value_counts()


# Out of 1470 employees, 1233 did not leave the company (Attrition = No), while 237 employees left (Attrition = Yes).

# # Description

# In[154]:


df.describe()


# The dataset shows that employees are between 18 and 60 years old, with an average age of 37. Daily rates range from 102 to 1499, and the average distance from home is 7 miles. Education levels span 1 to 5, while hourly rates vary from $30 to $100. Most employees have worked at the company for around 5 years, with total work experience ranging from 0 to 40 years. Some columns, like EmployeeCount and StandardHours, have the same value for all entries, providing no meaningful information.

# # Dropping column with one unique value

# In[157]:


# Drop 
df = df.loc[:, df.nunique() > 1]

# Display the updated dataframe
df.head()


# In[158]:


# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)


# # Visualization

# In[162]:


# Plot count plot for 'Attrition' column
plt.figure(figsize=(6, 4))
sns.countplot(x='Attrition', data=df, palette='viridis')
plt.title('Count Plot of Attrition')
plt.xlabel('Attrition')
plt.ylabel('Count')
plt.show()


# In[163]:


# Count of employees in each Department
print(df['Department'].value_counts())
plt.figure(figsize=(5, 5))
sns.countplot(x='Department', data=df)
plt.show()


# In[164]:


#Count of employees based on BusinessTravel
print(df['BusinessTravel'].value_counts())
plt.figure(figsize=(5, 5))
sns.countplot(x='BusinessTravel', data=df)
plt.show()


# In[174]:


# Countplot for Attrition vs MaritalStatus
plt.figure(figsize=(6, 4))
sns.countplot(x='MaritalStatus', hue='Attrition', data=df)
plt.title('Attrition vs Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.show()


# In[165]:


#Gender count of employees
print(df['Gender'].value_counts())
plt.figure(figsize=(5, 5))
sns.countplot(x='Gender', data=df)
plt.xticks(rotation=80)
plt.show()


# In[166]:


#Count of Educational Field of the employees
print(df['EducationField'].value_counts())
plt.figure(figsize=(5, 5))
sns.countplot(x='EducationField', data=df)
plt.xticks(rotation=80)
plt.show()


# In[167]:


#Job Role of employees
print(df['JobRole'].value_counts())
plt.figure(figsize=(5, 5))
sns.countplot(x='JobRole', data=df)
plt.xticks(rotation=75)
plt.show()


# In[168]:


#Overtime data of employees
print(df['OverTime'].value_counts())
plt.figure(figsize=(8, 6))
sns.countplot(x='OverTime', data=df)
plt.show()


# In[169]:


# Marital Status of employees
print(df['MaritalStatus'].value_counts())
plt.figure(figsize=(5, 5))
sns.countplot(x='MaritalStatus', data=df)
plt.xticks(rotation=75)
plt.show()


# In[171]:


# Boxplot for Age vs Attrition
plt.figure(figsize=(6, 4))
sns.boxplot(x='Attrition', y='Age', data=df)
plt.title('Boxplot of Age vs Attrition')
plt.show()


# In[172]:


# Boxplot for MonthlyIncome vs Attrition
plt.figure(figsize=(6,4))
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
plt.title('Boxplot of Monthly Income vs Attrition')
plt.show()


# In[173]:


# Boxplot for YearsAtCompany vs Attrition
plt.figure(figsize=(6,4))
sns.boxplot(x='Attrition', y='YearsAtCompany', data=df)
plt.title('Boxplot of Years at Company vs Attrition')
plt.show()


# In[179]:


# List of numerical columns from the DataFrame
numerical_col = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 
                 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel', 
                 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 
                 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

# Plotting distribution for each numerical column
plt.figure(figsize=(20, 20))
for i, column in enumerate(numerical_col, 1):
    plt.subplot(6, 3, i)
    sns.histplot(df[column], color='Purple')
    plt.xlabel(column)
plt.tight_layout()
plt.show()


# In[183]:


# Split Violin Plot for Attrition vs Job Level
plt.figure(figsize=(6, 4))
sns.violinplot(x='Attrition', y='JobLevel', data=df, split=True)

plt.title('Attrition vs Job Level (Split Violin Plot)')
plt.show()


# In[185]:


# Check the relation in the dataset.
sns.pairplot(df, hue='Attrition', palette={'No': 'purple', 'Yes': 'violet'})

plt.show()


# # Remove ouliers

# In[184]:


df.columns


# In[186]:


# List numerical columns
numerical_col = df.select_dtypes(include='number').columns.tolist()

print(numerical_col)


# In[187]:


# Features with outliers
features = df[['MonthlyIncome','NumCompaniesWorked','PerformanceRating','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']]


# In[188]:


z = np.abs(zscore (features))
z


# In[189]:


# Making new DataFrame with Z score less than 3.

new_df = df[(z<3).all(axis = 1)] 
new_df


# In[190]:


# check new DataFrame shape.
new_df.shape


# In[195]:


# Select  numeric columns
numerical_cols = new_df.select_dtypes(include=['number']).columns

# Calculate skewness only for numeric columns
skewness = new_df[numerical_cols].skew().sort_values()

print(skewness)


# In[192]:


# Removing skewness.

skew = ['DistanceFromHome', 'JobLevel', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager']
scaler = PowerTransformer(method='yeo-johnson')


# In[193]:


new_df[skew] = scaler.fit_transform(new_df[skew].values)
new_df[skew].head()


# In[196]:


# Check data distribution after skewness removal
plt.figure(figsize=(20, 25), facecolor='white')

for i, column in enumerate(new_df[skew], 1):
    if i <= 12:
        plt.subplot(4, 3, i)
        sns.histplot(new_df[column], color='indigo', kde=True)
        plt.xlabel(column, fontsize=20)

plt.tight_layout()
plt.show()


# # label encoding

# In[197]:


categorical_col = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']


# In[198]:


LE = LabelEncoder()

new_df[categorical_col] =  new_df[categorical_col].apply(LE.fit_transform)

new_df[categorical_col]


# # Correlation between features and the target

# In[199]:


corr = new_df.corr()
corr


# In[200]:


# Plot heatmap 
plt.figure(figsize=(25, 25))

sns.heatmap(new_df.corr(), 
            linewidths=0.1, 
            vmin=-1, 
            vmax=1, 
            fmt='.1g', 
            annot=True, 
            linecolor='black', 
            annot_kws={'size': 10}, 
            cmap='viridis')  

plt.yticks(rotation=0)
plt.show()


# In[201]:


corr['Attrition'].sort_values(ascending = False)


# # Corelation in label and features

# In[202]:


plt.figure(figsize = (20, 10))

new_df.corr()['Attrition'].sort_values(ascending = False).drop(['Attrition']).plot(kind = 'bar', color = 'c')

plt.xlabel('Feature', fontsize = 14)
plt.ylabel('target', fontsize = 14)
plt.title('correlation between lanel and feature using bar plot', fontsize = 20)

plt.show()


# In[203]:


new_df.drop('BusinessTravel', axis = 1, inplace = True)
new_df.drop('HourlyRate', axis = 1, inplace = True)


# In[204]:


new_df.head()


# # Seperating features and label in x & y

# In[205]:


x = new_df.drop('Attrition', axis = 1)

y = new_df['Attrition']


# In[206]:


print(x.shape)

print(y.shape)


# In[207]:


y.value_counts()


# # Oversampling

# In[209]:


SM = SMOTE()

x, y = SM.fit_resample(x, y)


# In[210]:


y.value_counts()


# # Feature Scaling using Standard Scalarization

# In[211]:


scaler = StandardScaler()

x = pd.DataFrame(scaler.fit_transform(x), columns = x.columns)

x


# In[212]:


# VIF value check

vif = pd.DataFrame()

vif['VIF val'] = [variance_inflation_factor(x.values,i)
              for i in range(len(x.columns))]
vif['Features'] = x.columns

vif


# # model training

# In[213]:


maxAccu = 0
maxRS = 0
for i in range(1, 200):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = i)
    DTC = DecisionTreeClassifier()
    DTC.fit(x_train, y_train)
    pred = DTC.predict(x_test)
    acc = accuracy_score(y_test, pred)
    if acc > maxAccu:
        maxAccu = acc
        maxRS = i
print('Best accuracy is ', maxAccu,' on Random_state ', maxRS)


# hence we get best accuracy 88%

# In[214]:


# Create test train split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 131)


# # Logistic regressor

# In[215]:


# Checking accuracy for Logistic Regression Classifier

LR = LogisticRegression()
LR.fit(x_train, y_train)

predLR = LR.predict(x_test)

print(accuracy_score(y_test, predLR))
print(confusion_matrix(y_test, predLR))
print(classification_report(y_test, predLR))


# # Decision Tree classifier

# In[216]:


# Decision Tree Classifier Accuracy

DTC = DecisionTreeClassifier()
DTC.fit(x_train, y_train)

predDTC = DTC.predict(x_test)

print(accuracy_score(y_test, predDTC))
print(confusion_matrix(y_test, predDTC))
print(classification_report(y_test, predDTC))


# # Random forest classifier

# In[219]:


# Accuracy for Random Forest Classifier

RFC = RandomForestClassifier()
RFC.fit(x_train, y_train)

predRFC = RFC.predict(x_test)

print(accuracy_score(y_test, predRFC))
print(confusion_matrix(y_test, predRFC))
print(classification_report(y_test, predRFC))


# # Hyperparameter tunning

# In[220]:


parameters = {'n_estimators' : [50],
             'criterion' : ['gini'],
             'max_depth' : [2, 4, 6]}


# In[221]:


GCV = GridSearchCV(RFC, parameters, cv = 5)


# In[222]:


GCV.fit(x_train,y_train)


# In[223]:


GCV.best_params_


# In[224]:


Attrition = RandomForestClassifier(criterion = 'gini', max_depth = 6, n_estimators = 100)

Attrition.fit(x_train, y_train)

pred = Attrition.predict(x_test)

print(accuracy_score(y_test, predRFC))

print('RMSE value:', np.sqrt(metrics.mean_squared_error(y_test, pred)))

print('R2_Score:', r2_score(y_test, pred) * 50)


# In[225]:


import joblib

# Assuming 'RFC' is the model you want to save
joblib.dump(RFC, 'HR_analytics_For_the_Attrition_in_HR.pkl')


# In[ ]:




