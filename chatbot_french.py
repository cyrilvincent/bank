import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('data/chatbot/dialogs_fr.txt', sep='|')
df = df.dropna()

print(df.head())

def cleaner(x):
    return [a for a in (''.join([a for a in x if a not in [",", ".", "?", "!"]])).lower().split()]

model = Pipeline([
    ('bow', CountVectorizer(analyzer=cleaner)),
    ('tfidf', TfidfTransformer()),
    ('classifier', RandomForestClassifier())
])

model.fit(df.question, df.answer)

print("> Bonjour")
res = model.predict(['Bonjour'])[0]
print(res)

while True:
    s = input("> ")
    print(model.predict([s])[0])

