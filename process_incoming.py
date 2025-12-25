import pandas as pd
import requests
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from preprocess_json import create_embedding

df = joblib.load('embeddings.joblib')

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json = {
        "model" : "gemma3:4b",
        "prompt" : prompt,
        "stream" : False
    })

    response = r.json()
    print(response)
    print("\n\n--------------------------\n\n")
    return(response)

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]
# print(question_embedding)

# find similarities of question_embeddings with other embeddings
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
print(similarities)
top_results = 5
max_index = similarities.argsort()[::-1][0:top_results]
print(max_index)
new_df = df.loc[max_index]
# print(new_df[["title", "number", "text"]])

prompt = f'''I have some videos regarding NLP from which subtitles are extracted.
Here are video subtitle chunks containing video title, video number, start time in second, end time in seconds, the text at that time:
{new_df[["title", "number", "start", "end", "text"]].to_json(orient = "records")}
---------------------
{incoming_query}
User asked this question related to the video chunks, you have to answer where an how much content is taught where (in which video and at what timestamp) and guide the user to go to that particular video
If user asks unrelated question, tell him that you can only answer questions related to the course.
'''

with open("promt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)

for index, item in new_df.iterrows():
    print(index, item['title'], item['number'], item['text'], item['start'], item['end'])