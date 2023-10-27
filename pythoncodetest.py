import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from pythontest4.classifier import clf, y_pred

# Load the dataset
data = pd.read_csv("C:\\Users\\pp\\PycharmProjects\\Pythontest\\pythontest4\\SDS\\Car_Purchasing_Data.csv",encoding='ISO-8859-1')

# Summary statistics
print(data)
print(data.describe())
# Visualize data distributions
sns.histplot(data['Car Purchase Amount'], bins=20)
plt.show()

print("________________________Identify missing values_____________________")
print(data.isnull().sum())

data.plot(kind='hist')
plt.show()
sns.pairplot(data)
plt.show()

data.hist(figsize=(15,15))
plt.show()
value_count = data['Car Purchase Amount'].value_counts()
print(value_count)

def cat_valcount_hist(feature):
    print(data[feature].value_counts())
    fig, ax = plt.subplots( figsize = (6,6) )
    sns.countplot(x=feature, ax=ax, data=data)
    plt.show()


cat_valcount_hist('Gender')
cat_valcount_hist('Age')
cat_valcount_hist('Annual Salary')
cat_valcount_hist('Credit Card Debt')
cat_valcount_hist('Net Worth')

sns.boxplot(x='Car Purchase Amount',y='Net Worth',data=data)
plt.show()

x = data[['Gender','Age','Annual Salary','Credit Card Debt','Net Worth']]
y = data['Car Purchase Amount']

X_train,X_test,y_train,y_test = train_test_split(x,y,train_size=0.3,random_state=42)

fig,ax = plt.subplots(figsize=(8,8))
sns.countplot(x='Net Worth', hue='Car Purchase Amount',ax=ax,data=data)
plt.show()
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

# Visualize metrics and ROC curve
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}, ROC AUC: {roc_auc}")

fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Feature importance (if applicable)
feature_importance = pd.Series(clf.coef_[0], index=X_train.columns)
feature_importance.plot(kind='barh')

#####################################

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create and train the regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict on the testing set
y_pred = reg.predict(X_test)

# Model evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Visualize predictions and actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()

print(f"RMSE: {rmse}, R-squared: {r2}")

# Feature importance (if applicable)
# You can analyze feature coefficients if using linear regression.
