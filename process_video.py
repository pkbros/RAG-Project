# Converts the videos to mp3
# Uses ffmpeg program to do so

import os
import subprocess

files = os.listdir("videos") #Retriving all videos


for file in files:
    video_number = file.split("-")[0].split("Day ")[1]
    video_name = file.split("- ")[1].split(" Quiz")[0]
    print(video_number, video_name)

    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{video_number}_{video_name}.mp3"])