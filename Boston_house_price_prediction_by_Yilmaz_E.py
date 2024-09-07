# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
matplotlib.use('QT5Agg')
from matplotlib import pyplot
from matplotlib import pyplot as plt
import helperFunctions as hf

# To see all columns at once
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: '%.3f' % x)
pd.set_option("display.width", 500)

#####################################################################################################################
###################################### Loading and Examining the Dataset ############################################
#####################################################################################################################

df = pd.read_csv('Boston_house_prices.csv')
'''
1) CRIM: per capita crime rate by town
2) ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
3) INDUS: proportion of non-retail business acres per town
4) CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
5) NOX: nitric oxides concentration (parts per 10 million) [parts/10M]
6) RM: average number of rooms per dwelling
7) AGE: proportion of owner-occupied units built prior to 1940
8) DIS: weighted distances to five Boston employment centres
9) RAD: index of accessibility to radial highways
10) TAX: full-value property-tax rate per $10,000 [$/10k]
11) PTRATIO: pupil-teacher ratio by town
12) B: The result of the equation B=1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
13) LSTAT: % lower status of the population

Output variable:
1) MEDV: Median value of owner-occupied homes in $1000's [k$]
'''

# Examining the dataset
def check_data(dataframe,head=5):
    print(20*"-" + "Information".center(20) + 20*"-")
    print(dataframe.info())
    print(20*"-" + "Data Shape".center(20) + 20*"-")
    print(dataframe.shape)
    print("\n" + 20*"-" + "The First 5 Data".center(20) + 20*"-")
    print(dataframe.head())
    print("\n" + 20 * "-" + "The Last 5 Data".center(20) + 20 * "-")
    print(dataframe.tail())
    print("\n" + 20 * "-" + "Missing Values".center(20) + 20 * "-")
    print(dataframe.isnull().sum())
    print("\n" + 40 * "-" + "Describe the Data".center(40) + 40 * "-")
    print(dataframe.describe([0.01, 0.05, 0.10, 0.50, 0.75, 0.90, 0.95, 0.99]).T)
hf.check_data(df)
df.head(25)
'''
506 rows and 14 columns - one of if it is the target variable.
There is no missing data.
All float64-type variables except CHAS and RAD being int64.
'''

#####################################################################################################################

# Plot histograms for each feature
df.hist(bins=20, figsize=(15, 10))
plt.show()
'''
There are skewed distributions for --> CRIM, AGE, DIS, B, LSTAT.
There are several variables can be categorized --> ZN, INDUS, CHAS, NOX, RAD, TEX, PTRATIO.
Others have normal distribution.
'''

# Apply logarithmic transformation to specified columns
columns_to_transform = ['CRIM', 'AGE', 'DIS', 'B', 'LSTAT']

for column in columns_to_transform:
    # Add a small constant to avoid log(0), though log1p handles this by default
    df[column] = np.log1p(df[column])

# Display the first few rows of the DataFrame to check the transformations
print(df.head())

#####################################################################################################################

# Plot boxplots for each feature
plt.figure(figsize=(15, 10))
sns.boxplot(data=df, orient='h')
plt.show()
'''
High outlier numbers are seen for --> CRIM, ZN, B
There are outliers for CHAS, RM, DIS, PTRATIO, LSTAT, and MEDV but not as scatter as the variables just mentioned above.
'''

#####################################################################################################################

# Correlation matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
'''

'''

#####################################################################################################################

# Correlation of all the variables with just the target
# Compute the correlation matrix
corr_matrix = df.corr()

# Extract correlations with the target variable 'MEDV'
target_corr = corr_matrix['MEDV'].sort_values(ascending=False)

# Display the correlations with the target variable
print(target_corr)

# Visualize the correlations as a bar plot
plt.figure(figsize=(8, 6))

# Create a color palette based on the correlation values
colors = target_corr.apply(lambda x: 'red' if x < 0 else 'blue')

sns.barplot(x=target_corr.index, y=target_corr.values, palette=colors)
plt.title('Correlation with MEDV')
plt.xticks(rotation=45)
plt.ylabel('Correlation Coefficient')
plt.show()

#####################################################################################################################

# Segmentation of correlations seen in the heatmap and their report
# Compute the correlation matrix
corr_matrix = df.corr()

# Extract the upper triangle of the correlation matrix (excluding the diagonal)
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Flatten the upper triangle into a Series and drop NA values
flat_corr = upper_triangle.stack()

# Define segments
bins = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
         0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
labels = ['-1.0 to -0.9', '-0.9 to -0.8', '-0.8 to -0.7', '-0.7 to -0.6',
          '-0.6 to -0.5', '-0.5 to -0.4', '-0.4 to -0.3', '-0.3 to -0.2',
          '-0.2 to -0.1', '-0.1 to 0.0', '0.0 to 0.1', '0.1 to 0.2',
          '0.2 to 0.3', '0.3 to 0.4', '0.4 to 0.5', '0.5 to 0.6',
          '0.6 to 0.7', '0.7 to 0.8', '0.8 to 0.9', '0.9 to 1.0']

# Group the correlations into the defined segments
grouped_corr = pd.cut(flat_corr, bins=bins, labels=labels).value_counts().sort_index()

# Report the number of pairs in each segment
print(grouped_corr)

# Optionally, display the pairs that fall into each segment
for label in labels:
    print(f'\nCorrelations in segment {label}:')
    pairs_in_segment = flat_corr[(flat_corr > bins[labels.index(label)]) &
                                 (flat_corr <= bins[labels.index(label) + 1])]
    print(pairs_in_segment)

#####################################################################################################################

# Scatter plot between RM (average number of rooms per dwelling) and PRICE
for col in df.columns:
    if col != 'MEDV':
        plt.figure(figsize=(8, 6))
        plt.scatter(df[col], df['MEDV'])
        plt.xlabel(f'{col}')
        plt.ylabel('MEDV')
        plt.title(f'Relationship between {col} and MEDV')
        plt.show()

#####################################################################################################################

# Joint plot between LSTAT (percentage of lower status population) and PRICE
for col in df.columns:
    sns.jointplot(x=f'{col}', y='MEDV', data=df, kind='reg')
    plt.show()

#####################################################################################################################

# Distribution of the target variable (PRICE)
sns.histplot(df['MEDV'], bins=20, kde=True)
plt.xlabel('MEDV')
plt.ylabel('Frequency')
plt.title('Distribution of MEDV')
plt.show()

#####################################################################################################################
###################################### Outlier Handling #############################################################
#####################################################################################################################

# Box-and-whiskers plots of the variables to handle with outliers
for column in df.columns:
    plt.figure(figsize=(10, 5))
    df.boxplot(column=column, vert=False)
    plt.title(f'Box-and-Whisker Plot for {column}')
    plt.show()

#####################################################################################################################

# CRIM can be categorized into three as the ones in the interval of box-and-whisker, the outliers up to 25, and the rest
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['CRIM'].quantile(0.25)
Q3 = df['CRIM'].quantile(0.75)

# Calculate the Interquartile Range (IQR)
IQR = Q3 - Q1

# Calculate the lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

def categorize_crim(value):
    if value <= upper_bound:
        return 'A'  # Non-outliers
    elif value <= 25:
        return 'B'  # Outliers up to 25
    else:
        return 'C'  # Outliers above 25

# Apply the function to create a new column
df['CRIM_Category'] = df['CRIM'].apply(categorize_crim)

print(df[['CRIM', 'CRIM_Category']].head())

#####################################################################################################################

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['ZN'].quantile(0.25)
Q3 = df['ZN'].quantile(0.75)

# Calculate the Interquartile Range (IQR)
IQR = Q3 - Q1

# Calculate the lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

def categorize_zn(value):
    if value <= upper_bound and value >= lower_bound:
        return 'Non-outlier'  # Values within the non-outlier range
    else:
        return 'Outlier'  # Values outside the non-outlier range

# Apply the function to create a new column
df['ZN_Category'] = df['ZN'].apply(categorize_zn)

print(df[['ZN', 'ZN_Category']].head())

#####################################################################################################################

# Identify numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns

# Filter DataFrame to include only numerical columns
df_numerical = df[numerical_cols]

# Calculate Z-scores only for numerical columns
z_scores = np.abs((df_numerical - df_numerical.mean()) / df_numerical.std())

# Define a threshold to identify outliers
threshold = 3

# Identify outliers (returns a DataFrame with True for outliers)
outliers = z_scores > threshold

# Optionally, you can filter out the outliers for inspection
outlier_rows = df[outliers.any(axis=1)]
print(outlier_rows)

#####################################################################################################################

# Converting 'CHAS' to a categorical variable
df['CHAS'] = df['CHAS'].astype('category')

#####################################################################################################################

'''
def categorize_b(value):
    if value <= 100:
        return 'Low'  # 0 to 100
    elif value <= 275:
        return 'Moderate'  # 100 to 275
    elif value <= 350:
        return 'High'  # 275 to 350
    else:
        return 'Very High'  # Higher than 350

# Apply the function to create a new column
df['B_Category'] = df['B'].apply(categorize_b)

print(df[['B', 'B_Category']].head())
'''
#####################################################################################################################

# Identify numerical columns
numerical_col = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Identify categorical columns
categorical_col = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Print the lists to verify
print("Numerical Columns:", numerical_col)
print("Categorical Columns:", categorical_col)

#####################################################################################################################

# Subset DataFrame with only numerical columns
df_numerical = df[numerical_col]

# Compute the correlation matrix
correlation_matrix = df_numerical.corr()

plt.figure(figsize=(10, 8))

# Create the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Add titles and labels
plt.title('Correlation Heatmap of Numerical Features')
plt.xlabel('Features')
plt.ylabel('Features')

# Show the plot
plt.show()

#####################################################################################################################
###################################### Feature Engineering ##########################################################
#####################################################################################################################

df['RM_sqroot'] = df['RM']**2
df['LSTAT_sqroot'] = df['LSTAT']**2
df['ZN_times_NOX'] = df['ZN'] * df['NOX']
df['CRIM_times_LSTAT'] = df['CRIM'] * df['LSTAT']
df['DIS_times_RAD'] = df['DIS'] * df['RAD']
df['CRIM_times_INDUS'] = df['CRIM'] * df['INDUS']
df['AGE_squared'] = df['AGE']**1/2
df['CRIM_squared'] = df['CRIM']**2
df['ZN_times_PTRATIO'] = df['ZN'] * df['PTRATIO']

#####################################################################################################################
####################################### Encoding ####################################################################
#####################################################################################################################

# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Apply One-Hot Encoding
encoded_data = encoder.fit_transform(df[categorical_col])

# Convert to DataFrame
df_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_col))

# Concatenate with the original DataFrame
df_final = pd.concat([df.drop(columns=categorical_col), df_encoded], axis=1)

print(df_final)
df = df_final
df.head(25)

#####################################################################################################################
###################################### Training the Model ###########################################################
#####################################################################################################################

# Split the data into training and testing sets
X = df.drop('MEDV', axis=1)  # Features
y = df['MEDV']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#####################################################################################################################
###################################### Scaling ######################################################################
#####################################################################################################################

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#####################################################################################################################
###################################### Making Predictions ###########################################################
#####################################################################################################################

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the scaled training data
model.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred = model.predict(X_test_scaled)

#####################################################################################################################
###################################### Model Evaluation #############################################################
#####################################################################################################################

# Calculate Mean Squared Error and R^2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.title('Actual vs Predicted MEDV')
plt.show()

# Residuals
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='blue')
plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), color='red', linewidth=2)
plt.xlabel('Predicted MEDV')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted  MEDV')
plt.show()

# Assuming X and y are already defined as features and target
# and model is already trained

# Get the feature names from the DataFrame
feature_names = X.columns

# Get the coefficients from the trained model
coefficients = model.coef_

# Create a DataFrame for feature importances
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Sort the features by the absolute value of coefficients in descending order
feature_importance = feature_importance.reindex(
    feature_importance.Coefficient.abs().sort_values(ascending=False).index
)

# Generate a color map
colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))

# Plot the feature importances
plt.figure(figsize=(12, 8))
bars = plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors)
plt.xlabel('Coefficient Value')
plt.title('Feature Importance from Linear Regression Model')

# Optional: Add value labels on the bars
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}',
             va='center', ha='left', color='black')

plt.show()


X.head(25)









# Calculate the IQR for the MEDV column
Q1 = df['MEDV'].quantile(0.25)
Q3 = df['MEDV'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier criteria
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers_medv = df[(df['MEDV'] < lower_bound) | (df['MEDV'] > upper_bound)]

# Display rows with outliers in the MEDV column
print(outliers_medv)




import pandas as pd

# Assuming 'df' is your DataFrame and 'MEDV' is the column of interest

# Step 1: Filter out all rows where MEDV is 50
rows_with_50k = df[df['MEDV'] == 50]

# Step 2: Randomly select one of these rows to keep
row_to_keep = rows_with_50k.sample(n=1, random_state=42)

# Step 3: Filter out rows with MEDV = 50 from the main DataFrame
df_filtered = df[df['MEDV'] != 50]

# Step 4: Add the one row back to the filtered DataFrame
df_final = pd.concat([df_filtered, row_to_keep], ignore_index=True)
df = df_final

# Now 'df_final' is your DataFrame with all rows where MEDV = 50 removed except one random row
