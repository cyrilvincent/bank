import csv
import pickle
from typing import List
import numpy as np
import os
import tempfile
import tensorflow_hub as hub

print(os.path.join(tempfile.gettempdir(), "tfhub_modules"))

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #"https://tfhub.dev/google/universal-sentence-encoder-lite/2" #"https://tfhub.dev/google/universal-sentence-encoder/4" #"https://tfhub.dev/google/universal-sentence-encoder-large/5"
model = hub.load(module_url)
print ("module %s loaded" % module_url)

class Item:

    def __init__(self, question, answer):
        self.question = question
        self.answer = answer
        self.embed = None

class BertUseService:

    def __init__(self):
        self.items: List[Item] = []

    def embed(self, s: str):
        return model([s])[0].numpy()

    def score(self, l1: List[float], l2: List[float]) -> float:
        return np.inner(l1, l2)

    def train(self, path: str):
        print("Training")
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                q = row["question"]
                r = row["answer"]
                if r is not None and r.strip() != "":
                    item = Item(q, r)
                    item.embed = self.embed(q)
                    self.items.append(item)
        print("\nSaving")
        with open(path.replace(".txt", "_model.pickle"), "wb") as f:
            pickle.dump(self.items, f)

    def load_model(self, path):
        print("Load model")
        with open(path, "rb") as f:
            self.items = pickle.load(f)

    def predict(self, s: str):
        embed = self.embed(s)
        best_score = 0
        best_item = None
        for item in self.items:
            score = self.score(embed, item.embed)
            if score > best_score:
                best_score = score
                best_item = item
                if best_score > 0.99:
                    break
        return best_item, best_score

if __name__ == '__main__':
    service = BertUseService()
    service.train("data/chatbot/dialogs_fr.txt")
    service.load_model("data/chatbot/dialogs_fr_model.pickle")
    print("> Bonjour")
    res = service.predict("Bonjour")
    print(f"{res[0].answer} @{res[1] * 100:.0f}%")
    while True:
        s = input("> ")
        res = service.predict(s)
        print(f"{res[0].answer} @{res[1] * 100:.0f}%")






