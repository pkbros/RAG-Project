import os
import whisper
import json

model = whisper.load_model("turbo")
audios = os.listdir("audios")

for audio in audios:
    # print(audio)
    number = audio.split("_")[0]
    title = audio.split("_")[1][:-4]


    result = model.transcribe(audio=f"audios/{audio}")


    chunks = []
    for segment in result["segments"]:
        chunks.append(
            {
                "number": number,
                "title": title,
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": segment["text"],
            }
        )
        chunks_with_metadata = {"chunks":chunks, "text": result["text"]}


    with open(f"jsons/{audio}.json","w") as f:
        json.dump(chunks_with_metadata, f)
    print(f"{audio} done ✌️")