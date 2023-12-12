import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

pd.options.display.max_columns = None
df = pd.read_csv("data/credit_risk_loan/loan.csv")
print(df.head())
print(df.describe())
print(df.info())

# Get only colomns nedeed
columns = ['loan_amnt','funded_amnt_inv',
           'term','int_rate','installment','grade','emp_length','home_ownership'
           ,'annual_inc','verification_status','purpose', 'loan_status','total_pymnt']

df = df[columns]
print(df.tail())

# Null
print(df.isnull().sum())
# Since we can't drop null value to annual_inc, otherwise I change fill with 0 value
df.loc[:, 'annual_inc'] = df.loc[:, 'annual_inc'].fillna(0)

# Discretisation
print(df['loan_status'].unique())

# Define a function to classify loan status into 0 (safe) or 1 (risky)
def classify_credit_risk(loan_status):
    risky_statuses = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period',
                      'Late (31-120 days)', 'Late (16-30 days)']
    if loan_status in risky_statuses:
        return 1  # Risky
    else:
        return 0  # Safe

# Create the 'class' column using the function and .loc
df['class'] = df['loan_status'].apply(classify_credit_risk)
print(df[['loan_status', 'class']])
df = df.drop('loan_status', axis=1)
df.reset_index(drop=True)

# EDA
# Exploraton Data analysis
sns.histplot(df['loan_amnt'])
plt.title('Distribution of Loan Amount', fontsize=14)
plt.show()

sns.countplot(x='class', data=df)
plt.title('Class Distributions \n (0: Safe || 1: Risk)', fontsize=14)
plt.show()

# Encoder
def apply_label_encoding(column):
    labelencoder = LabelEncoder()
    return labelencoder.fit_transform(column)

columns_to_encode = ['term','grade', 'emp_length', 'home_ownership','purpose','verification_status']
df[columns_to_encode] = df[columns_to_encode].apply(apply_label_encoding)
print(df.head())

# Correlation
target_correlations = df.corr()['class'].apply(abs).sort_values()
print(target_correlations)

y = df['class']
x = df.drop('class', axis=1)

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
x_scaled = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Réequilibrage
from imblearn.combine import SMOTETomek
smote = SMOTETomek()
x_train, y_train = smote.fit_resample(x_train, y_train)
from collections import Counter
print("The number of classes before fit {}".format(Counter(y_train)))

print("Fitting")
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

print(classification_report(y_test,y_pred))

from sklearn.metrics import roc_curve

lr_auc = roc_auc_score(y_test, y_pred)
# summarize scores
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)
# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

