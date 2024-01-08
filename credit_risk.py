import re
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from feature_engine.selection import DropConstantFeatures, DropCorrelatedFeatures, DropDuplicateFeatures
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, \
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from yellowbrick.classifier import ClassPredictionError

df = pd.read_csv("data/credit_risk/credit_risk_dataset.csv")
print(df.head())
df = df.drop_duplicates()
df = df.dropna()

def grab_col_names(dataframe):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"] #O = object

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and # Numérique mais cat car < 10 values
                   dataframe[col].dtypes != "O"]
    cat_cols = cat_cols + num_but_cat
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols

cat_cols, num_cols = grab_col_names(df)


def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    # Select only the numeric columns from the DataFrame
    numeric_dataframe = dataframe.select_dtypes(include=['number'])

    corr = numeric_dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

    if plot:
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr * 100, annot=True, fmt='.2f', mask=mask)
        plt.title('Confusion Matrix')
        plt.show()

    return drop_list


# Usage example:
high_correlated_cols(df, plot=True)

# y distribution
temp=dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=12),
                           height=500, width=1000))
target=df.loan_status.value_counts(normalize=True)
target.rename(index={1:'Default',0:'non default'},inplace=True)
pal, color=['#016CC9','#DEB078'], ['#8DBAE2','#EDD3B3']
fig=go.Figure()
fig.add_trace(go.Pie(labels=target.index, values=target*100, hole=.45,
                     showlegend=True,sort=False,
                     marker=dict(colors=color,line=dict(color=pal,width=2.5)),
                     hovertemplate = "%{label} Accounts: %{value:.2f}%<extra></extra>"))
fig.update_layout(template=temp, title='Target Distribution',
                  legend=dict(traceorder='reversed',y=1.05,x=0),
                  uniformtext_minsize=15, uniformtext_mode='hide',width=700)
fig.show()

# Distribution
import plotly.tools as tls
import plotly.offline as py
df_good = df.loc[df["loan_status"] == 1]['person_age'].values.tolist()
df_bad = df.loc[df["loan_status"] == 0]['person_age'].values.tolist()
df_age = df['person_age'].values.tolist()

#First plot
trace0 = go.Histogram(
    x=df_good,
    histnorm='probability',
    name="Loan status = 1"
)
#Second plot
trace1 = go.Histogram(
    x=df_bad,
    histnorm='probability',
    name="Loan status = 0"
)
#Third plot
trace2 = go.Histogram(
    x=df_age,
    histnorm='probability',
    name="Overall Age"
)

#Creating the grid
fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                          subplot_titles=('Good','Bad', 'General Distribuition'))

#setting the figs
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)

fig['layout'].update(showlegend=True, title='Age Distribuition', bargap=0.05)
py.iplot(fig, filename='custom-sized-subplot-with-subplot-titles')

# Housing distribtion
trace0 = go.Bar(
    x = df[df["loan_status"]== 1]["person_home_ownership"].value_counts().index.values,
    y = df[df["loan_status"]== 1]["person_home_ownership"].value_counts().values,
    name='Loan status = 1'
)

#Second plot
trace1 = go.Bar(
    x = df[df["loan_status"]== 0]["person_home_ownership"].value_counts().index.values,
    y = df[df["loan_status"]== 0]["person_home_ownership"].value_counts().values,
    name="Loan status = 0"
)

data = [trace0, trace1]

layout = go.Layout(
    title='Housing Distribuition'
)


fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='Housing-Grouped')

# Loan Grade
trace0 = go.Bar(
    x = df[df["loan_status"]== 1]["loan_grade"].value_counts().index.values,
    y = df[df["loan_status"]== 1]["loan_grade"].value_counts().values,
    name='Loan status = 1'
)

#Second plot
trace1 = go.Bar(
    x = df[df["loan_status"]== 0]["loan_grade"].value_counts().index.values,
    y = df[df["loan_status"]== 0]["loan_grade"].value_counts().values,
    name="Loan status = 0"
)

data = [trace0, trace1]

layout = go.Layout(
    title='Loan grade'
)


fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='Loan grade')

# Discretisation de personne_income
df['income_group'] = pd.cut(df['person_income'],
                              bins=[0, 25000, 50000, 75000, 100000, float('inf')],
                              labels=[0, 1, 2, 3, 4])

# make dataset
cat_cols.remove("loan_status")
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int) # Passage en to_categorical
x = df.drop(['loan_status',"person_age","person_income"], axis=1)
y = df['loan_status']


pipeline = Pipeline(steps=[
    ('constant',DropConstantFeatures()),
    ('correlated',DropCorrelatedFeatures()),
    ('duplicate',DropDuplicateFeatures())
])

x = pipeline.fit_transform(x)
print(x.shape)

from sklearn.model_selection import train_test_split
np.random.seed(42)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, stratify=y)
scaler = RobustScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)


models = []
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_auc_scores = []
def train_and_evaluate_model(model):
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    print("Classification Report:")
    print(classification_report(ytest, ypred))
    print('-' * 50)
    ConfusionMatrixDisplay.from_predictions(ytest, ypred)
    PrecisionRecallDisplay.from_predictions(ytest, ypred)
    RocCurveDisplay.from_predictions(ytest, ypred)
    acc = accuracy_score(ytest, ypred)
    precision = precision_score(ytest, ypred, average='macro')
    recall = recall_score(ytest, ypred, average='macro')
    f1 = f1_score(ytest, ypred, average='macro')
    roc_auc = roc_auc_score(ytest, ypred, average='macro') #ROC curves typically feature true positive rate (TPR) on the Y axis, and false positive rate (FPR) on the X axis. This means that the top left corner of the plot is the “ideal” point - a FPR of zero, and a TPR of one. This is not very realistic, but it does mean that a larger area under the curve (AUC) is usually better. The “steepness” of ROC curves is also important, since it is ideal to maximize the TPR while minimizing the FPR.

    if re.search('catboost', str(model)) == None:
        visualizer = ClassPredictionError(model)
        visualizer.score(xtest, ytest)
        visualizer.show()

    accuracy_scores.append(acc)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    roc_auc_scores.append(roc_auc)
    models.append(model)

from sklearn.ensemble import RandomForestClassifier
train_and_evaluate_model(RandomForestClassifier())
train_and_evaluate_model(XGBClassifier())



