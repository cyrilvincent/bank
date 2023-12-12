import spacy

# install : python -m spacy download fr_core_news_sm (lg for large) >=python 3.10


nlp = spacy.load("fr_core_news_sm")

with open("data/openai/bank.txt") as f:
        text = f.read()
doc = nlp(text)

print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
