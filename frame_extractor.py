#%%
import os
import glob
from tqdm import tqdm

DATA_PATH = '/data/common/abb_project/resized'
SAVE_PATH = '/data/common/abb_project/frames'

videos = glob.glob(os.path.join(DATA_PATH, '**', '*.mp4'), recursive=True)
error_files = set()
with open('processed_data/flow_error_files.txt', 'r') as f:
    for line in f:
        error_files.add(line.strip())
for video_file in tqdm(videos, desc='Extracting videos'):
    frames_folder_name = "__".join(video_file.split('/')[5:]).replace('.mp4', '')
    if video_file in error_files:
        continue
    if not os.path.exists(os.path.join(SAVE_PATH, frames_folder_name)):
        os.makedirs(os.path.join(SAVE_PATH, frames_folder_name))
        # extract frames with ffmpeg 30 fps quality 2 for jpg
        command = 'ffmpeg -i "{}" -vf fps=30 "{}/%06d.jpg"'.format(video_file, os.path.join(SAVE_PATH, frames_folder_name))
        os.system(command)