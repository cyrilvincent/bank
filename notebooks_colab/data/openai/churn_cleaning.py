import pandas as pd
import sweetviz
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_columns = None

df = pd.read_csv('data/churn/churn.csv', delimiter=',')
print(df.shape)

# Anonymisation
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)

# France only
df = df[df.Geography == "France"]
df = df.drop(["Geography"], axis = 1)
print(df.describe().T)

# my_report = sweetviz.analyze(df)
# my_report.show_html()

# Discretisation?
print(df.nunique())

# Etude pragmatique sur valeurs discretes (categorielles)
fig, axarr = plt.subplots(2, 2)
sns.countplot(x='Gender', hue = 'Exited',data = df, ax=axarr[0][0])
sns.countplot(x='HasCrCard', hue = 'Exited',data = df, ax=axarr[0][1])
sns.countplot(x='IsActiveMember', hue = 'Exited',data = df, ax=axarr[1][0])
plt.show()

# Etude pragmatique sur valeurs continues
fig, axarr = plt.subplots(3, 2)
sns.boxplot(y='CreditScore',x = 'Exited', hue = 'Exited',data = df, ax=axarr[0][0])
sns.boxplot(y='Age',x = 'Exited', hue = 'Exited',data = df , ax=axarr[0][1])
sns.boxplot(y='Tenure',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][0])
sns.boxplot(y='Balance',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][1])
sns.boxplot(y='NumOfProducts',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][0])
sns.boxplot(y='EstimatedSalary',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][1])
plt.show()

# Ingenieurie de la data
df['BalanceSalaryRatio'] = df.Balance/df.EstimatedSalary
df['TenureByAge'] = df.Tenure/(df.Age) # Tenure = mandat
df['CreditScoreGivenAge'] = df.CreditScore/(df.Age)

# Reordonne pour avoir les valeurs discretes à droite
continuous_vars = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio',
                   'TenureByAge','CreditScoreGivenAge']
cat_vars = ['HasCrCard', 'IsActiveMember','Gender']
df = df[['Exited'] + continuous_vars + cat_vars]
print(df.head())

df.loc[df.Gender == "Male", 'Gender'] = 1
df.loc[df.Gender == "Female", 'Gender'] = 0
print(df.head())

# Equilibrage
# Proportion de churn
labels = 'Exited', 'Retained'
sizes = [df.Exited[df.Exited==1].count(), df.Exited[df.Exited==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customer churned and retained", size = 20)
plt.show()

# Equilibrage
nb = len(df[df.Exited==1])
df = pd.concat([df[df.Exited==1], df[df.Exited==0][:nb]])

df.to_csv("data/churn/churn_clean.csv", index=False)
