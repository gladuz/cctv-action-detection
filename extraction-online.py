"""
extraction-online.py does the following:

1. Loads a dataset.pkl in processed_data folder.
2. and then get the videos' paths from the dataset.pkl.
3. and then concat the videos' paths with the root path of the dataset.
4. and then also put .mp4 extension to the videos' paths.
5. finally you got abs paths of the videos.

6. and then load the model.
7. and then load the video.
8. does some stuffs to the video.

the only difference between extraction-online.py and extraction-offline.py is that
extraction-online.py loads the video (RAM, and it uses only single temp video) while extraction-offline.py loads the video from
pre-saved entire .mp4 files.
"""

import pickle
import numpy as np
import os

with open('processed_data/dataset.pkl', 'rb') as f:
    data = pickle.load(f)
    
