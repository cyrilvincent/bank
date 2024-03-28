from openai import OpenAI

with open("data/openai/openai.env") as f:
    key = f.read()

with open("churn_rf.py") as f:
    text = f.read()

#print(text)

client = OpenAI(api_key=key)

completion = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Ajoute des commentaires docstrings dans ce code Python"},
        {"role": "user", "content": text}
    ]
)

res = completion.choices[0].message
print(res.content)
