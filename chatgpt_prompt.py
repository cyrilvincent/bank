from openai import OpenAI

def chat(system: str, user: str) -> str:
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return completion.choices[0].message.content

with open("data/openai/openai.env") as f:
    key = f.read()
client = OpenAI(api_key=key)
chat_model = "gpt-3.5-turbo-1106"
old_system = ""
old_user = ""
print("CTRL + C pour arrêter")
while True:
    system = input(f'System (Entrée pour "{old_system}") > ')
    if system == "":
        system = old_system
    user = input(f'> (Entrée pour "{old_user}") > ')
    if user == "":
        user = old_user
    s = chat(system, user)
    print(s)
    old_system = system
    old_user = user




