import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import xgboost

pd.options.display.max_columns = None
df = pd.read_csv('data/marketing/bank.csv')
print(df.head())

# y = deposit_bool

# Categorical
cat_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

fig, axs = plt.subplots(3, 3, sharex=False, sharey=False)
counter = 0
for cat_column in cat_columns:
    value_counts = df[cat_column].value_counts()
    trace_x = counter // 3
    trace_y = counter % 3
    x_pos = np.arange(0, len(value_counts))
    axs[trace_x, trace_y].bar(x_pos, value_counts.values, tick_label=value_counts.index)
    axs[trace_x, trace_y].set_title(cat_column)
    for tick in axs[trace_x, trace_y].get_xticklabels():
        tick.set_rotation(90)
    counter += 1
plt.show()

# Numerical
num_columns = ['balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
fig, axs = plt.subplots(2, 3, sharex=False, sharey=False)
counter = 0
for num_column in num_columns:
    trace_x = counter // 3
    trace_y = counter % 3
    axs[trace_x, trace_y].hist(df[num_column])
    axs[trace_x, trace_y].set_title(num_column)
    counter += 1
plt.show()

# Outliers
print(df[['pdays', 'campaign', 'previous']].describe())
print(len (df[df['pdays'] > 400] ) / len(df)) # 1.2%
# -1 possibly means that the client wasn't contacted before or stands for missing data.
print(len (df[df['campaign'] > 34] ) / len(df)) # 0.03%
print(len (df[df['previous'] > 34] ) / len(df)) # 0.04%

# Cleaning
def get_dummy_from_bool(row, column_name):
    ''' Returns 0 if value in column_name is no, returns 1 if value in column_name is yes'''
    return 1 if row[column_name] == 'yes' else 0


def get_correct_values(row, column_name, threshold, df):
    ''' Returns mean value if value in column_name is above threshold'''
    if row[column_name] <= threshold:
        return row[column_name]
    else:
        mean = df[df[column_name] <= threshold][column_name].mean()
        return mean


def clean_data(df):
    '''
    INPUT
    df - pandas dataframe containing bank marketing campaign dataset

    OUTPUT
    df - cleaned dataset:
    1. columns with 'yes' and 'no' values are converted into boolean variables;
    2. categorical columns are converted into dummy variables;
    3. drop irrelevant columns.
    4. impute incorrect values
    '''

    cleaned_df = df.copy()

    # convert columns containing 'yes' and 'no' values to boolean variables and drop original columns
    bool_columns = ['default', 'housing', 'loan', 'deposit']
    for bool_col in bool_columns:
        cleaned_df[bool_col + '_bool'] = df.apply(lambda row: get_dummy_from_bool(row, bool_col), axis=1)

    cleaned_df = cleaned_df.drop(columns=bool_columns)

    # convert categorical columns to dummies
    cat_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

    for col in cat_columns:
        cleaned_df = pd.concat([cleaned_df.drop(col, axis=1),
                                pd.get_dummies(cleaned_df[col], prefix=col, prefix_sep='_',
                                               drop_first=True, dummy_na=False)], axis=1)

    # drop irrelevant columns
    cleaned_df = cleaned_df.drop(columns=['pdays'])

    # impute incorrect values and drop original columns
    cleaned_df['campaign_cleaned'] = df.apply(lambda row: get_correct_values(row, 'campaign', 34, cleaned_df), axis=1)
    cleaned_df['previous_cleaned'] = df.apply(lambda row: get_correct_values(row, 'previous', 34, cleaned_df), axis=1)

    cleaned_df = cleaned_df.drop(columns=['campaign', 'previous'])

    return cleaned_df

cleaned_df = clean_data(df)
print(cleaned_df.head())

cleaned_df.to_csv("data/marketing/clean.csv", index=False)

# Training
x = cleaned_df.drop(columns = 'deposit_bool')
y = cleaned_df[['deposit_bool']]

scaler = pp.RobustScaler()
scaler.fit(x.values)
x_original = x
x = scaler.transform(x.values)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, test_size = 0.2, random_state=0)

xgb = xgboost.XGBClassifier(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
xgb.fit(xtrain,ytrain.squeeze().values)
print(f"Score {xgb.score(xtest, ytest)}")
with open("data/marketing/bank_xgb.pickle", "wb") as f:
    pickle.dump((scaler, xgb), f)

#get feature importances from the model
headers = ["name", "score"]
values = sorted(zip(x_original.columns, xgb.feature_importances_), key=lambda x: x[1] * -1)
xgb_feature_importances = pd.DataFrame(values, columns = headers)

#plot feature importances
x_pos = np.arange(0, len(xgb_feature_importances))
plt.bar(x_pos, xgb_feature_importances['score'])
plt.xticks(x_pos, xgb_feature_importances['name'])
plt.xticks(rotation=45)
plt.title('Feature importances (XGB)')

plt.show()

# Quantile
# Find out account balance, which marketing campaign should focus on:
df_new = cleaned_df.copy()
df_new['balance_deciles'] = pd.qcut(df_new['balance'], 10, labels=False, duplicates = 'drop')
#group by 'balance_buckets' and find average campaign outcome per balance bucket
mean_deposit = df_new.groupby(['balance_deciles'])['deposit_bool'].mean()

print(mean_deposit)
plt.plot(mean_deposit.index, mean_deposit.values)
plt.title('Mean % subscription depending on account balance')
plt.xlabel('balance deciles')
plt.ylabel('% subscription')
plt.show()

# Commence à être rentable au 6ème décile
print(df_new[df_new['balance_deciles'] == 6]['balance'].min())
# Le conseiller doit regarder à partir de 863€ de solde
print(df_new[df_new['balance_deciles'] == 9]['balance'].min())
# Dernier decile à 3899€





