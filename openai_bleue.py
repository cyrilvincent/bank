from openai import OpenAI

with open("data/openai/openai.env") as f:
    key = f.read()

with open("data/openai/quelle_est_bleue.txt") as f:
    text = f.read()

print(text)

client = OpenAI(api_key=key)

completion = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "De quoi parle ce texte?"},
        {"role": "user", "content": text}
    ]
)

res = completion.choices[0].message
print(res.content)
