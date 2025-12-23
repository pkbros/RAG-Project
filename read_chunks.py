# intall ollama(directly form website) and ollama's bge-m3(use ollama pull bge-m3 in terminal after install)

import requests
import json
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
    break


# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)
print(df)

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]
# print(question_embedding)

# find similarities of question_embeddings with other embeddings
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
print(similarities)
top_results = 3
max_index = similarities.argsort()[::-1][0:top_results]
print(max_index)
new_df = df.loc[max_index]
print(new_df[["title", "number", "text"]])