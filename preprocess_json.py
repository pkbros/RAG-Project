# intall ollama(directly form website) and ollama's bge-m3(use ollama pull bge-m3 in terminal after install)

import requests
import json
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model":"nomic-embed-text",
        "input": text_list
    })
    # print(r.json())
    # print(json.dumps(r, indent=4))


    embedding = r.json()["embeddings"]
    return embedding

# a = create_embedding("cat")
# print(a)


if __name__ == "__main__":
    jsons = os.listdir("jsons") #list all the jsons
    # print(jsons)
    my_dicts = []
    chunk_id = 0

    for json_file in jsons:
        
        with open(f"jsons/{json_file}", "r") as f:
            content = json.load(f)
        print(f"Creating embeddings for {json_file}")
        embeddings = create_embedding([c["text"] for c in content["chunks"]])
        
        for i,chunk in enumerate(content["chunks"]):
            chunk["chunk_id"] = chunk_id
            chunk_id += 1
            chunk["embedding"] = embeddings[i]
            my_dicts.append(chunk)



    # print(my_dicts)

    df = pd.DataFrame.from_records(my_dicts)
    # print(df)

    # Save the DataFrame
    joblib.dump(df, "embeddings.joblib")
