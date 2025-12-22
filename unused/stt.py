import whisper
import json

model = whisper.load_model("turbo")
result = model.transcribe("output.mp3")
# print(result)
print("\n\n-------------------------------\n")

chunks = []
for segment in result["segments"]:
    #converting to float makes it json serializable bcz by default all numbers are np.float32 not default python floats or int
    chunks.append({"start": float(segment["start"]), "end": float(segment["end"]), "text": segment["text"]})

# print(chunks)

with open("output.json", "w") as f:
    json.dump(chunks, f)
