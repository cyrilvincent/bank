import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('data/chatbot/dialogs.txt', sep='\t')

print(df.head())

def cleaner(x):
    return [a for a in (''.join([a for a in x if a not in [",", ".", "?", "!"]])).lower().split()]

model = Pipeline([
    ('bow', CountVectorizer(analyzer=cleaner)),
    ('tfidf', TfidfTransformer()),
    ('classifier', DecisionTreeClassifier())
])

model.fit(df.question, df.answer)

res = model.predict(['Hi'])[0]
print(res)
res = model.predict(["i've been good"])[0]
print(res)
