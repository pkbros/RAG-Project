# how to use this RAG AI Teaching assistant on your own data

## Step 1 - Collect your videos

Move all your video files to the videos folder

## Step 2 - convert to mp3

convert all the video files to mp3 by running video_to_mp3

## Step 3 - convert mp3 to json

convert all the mp3 files to json by running mp3_to_json

## Step 4 - convert the json to vectors

Use the file preprocess_json to convert the json files to a dataframe with embeddings

## Step 5 - Prompt generation and feeding to LLM

Read the joblib file and load into memory. Then create a relavant prompt as per the user query and feed it to the LLM
