import datetime
import whisper # pip install openai-whisper
import os
import numpy as np
from openai import OpenAI

class OpenAIService:

    def __init__(self, whisper_model="base", chat_model="gpt-3.5-turbo-1106", chat_limit=16385):
        os.environ["path"] += ";c:\\ffmpeg\\bin"
        self.model = whisper.load_model(whisper_model)
        with open("data/openai/openai.env") as f:
            key = f.read()
        self.client = OpenAI(api_key=key)
        self.chat_model = chat_model
        self.chat_limit = chat_limit

    def mp3_to_text(self, path):
        result = self.model.transcribe(path)
        return result["text"].encode("utf-8").decode()

    def summary(self, text: str, nb: int=5) -> str:
        completion = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "system", "content": f"Fais moi un résumé en {nb} parties de ce texte"},
                {"role": "user", "content": text}])
        return completion.choices[0].message.content

    def correction(self, text: str, ponctuation=True, bank=True) -> str:
        s = "Corriges moi ce texte "
        if ponctuation:
            s+="avec de la ponctuation "
        if bank:
            s+="avec un vocabulaire bancaire"
        completion = self.client.chat.completions.create(
            model=self.chat_model,
            # response_format={"type": "json_object"},
            messages=[{"role": "system", "content": s},
                      {"role": "user", "content": text}])
        return completion.choices[0].message.content


if __name__ == '__main__':
    print("OpenAI test")
    openai = OpenAIService(whisper_model="base")
    time0 = datetime.datetime.now()
    # res = openai.mp3_to_text("data/openai/bank.mp3")
    # print(res.strip())
    # with open("data/openai/bank.txt", "w") as f:
    #     f.write(res)

    with open("data/openai/bank.txt", "r") as f:
        res = f.read()
    correction = openai.correction(res)
    print(correction)
    with open(f"data/openai/correction.txt", "w") as f:
        f.write(correction)

    summary = openai.summary(res)
    print(summary)
    with open(f"data/openai/summary5.txt", "w") as f:
        f.write(summary)

    print(datetime.datetime.now() - time0)








