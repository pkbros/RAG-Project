# intall ollama(directly form website) and ollama's bge-m3(use ollama pull bge-m3 in terminal after install)

import requests
import json
import os

def create_embedding(text):
    r = requests.post("http://localhost:11434/api/embeddings", json={
        "model":"bge-m3",
        "prompt": text
    })
    # print(r.json())

    embedding = r.json()["embedding"]
    return embedding

# a = create_embedding("cat")
# print(a)

jsons = os.listdir("jsons")

for json_file in jsons:

    with open(f"jsons/{json_file}", "r") as f:
        content = json.load(f)
    for chunk in content["chunks"]:
        print(chunk)
    break