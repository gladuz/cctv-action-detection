# %%
import glob
import xml.etree.ElementTree as ET
import os
from xml.sax import SAXParseException
from xml.parsers.expat import ExpatError
import untangle
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm

DATA_PATH = '/data/common/abb_project/xmls'
VIDEO_PATH_NAME = 'resized'
FRAMES_PATH = '/data/common/abb_project/frames'
EVERY_N_FRAMES = 6
# ACTION_LABELS = ('normal', 'around', 'pushing', 'stop and go', 'pulling', 'kicking', 'throwing', 'piercing', 'punching', 'threaten', 'falldown')
ACTION_LABELS = ('piercing', 'stop and go', 'punching', 'threaten', 'kicking', 'running', 'pulling', 'around', 'climbwall', 'pushing', 'throwing', 'walking', 'falldown')
EVENT_LABELS = ('normal', 'assault', 'wander', 'trespass')

# each action has objectname, start, end, actionname
def parse_actions(ann_obj):
    """
    Return a list of actions
    [{
        'objectname': 'person',
        'actionname': 'assault',
        'start': 0,
        'end': 100
    },]
    """
    actions = []
    for person_objects in ann_obj.object:
        objectname = person_objects.objectname.cdata
        for action in person_objects.action:
            actionname = action.actionname.cdata
            for frame in action.frame:
                start = int(frame.start.cdata)
                end = int(frame.end.cdata)
                actions.append({
                    'objectname': objectname,
                    'actionname': actionname,
                    'start': start,
                    'end': end
                })
    return actions

def time_to_seconds(time):
    return sum(x * float(t) for x, t in zip([1, 60, 3600], reversed(time.split(":"))))


def parse_events(ann_obj):
    """
    Returns the list of big actions
    [{
        'actionname': 'assault',
        'start': 0,
        'end': 100
    },]
    """
    events = []
    for event in ann_obj.event:
        actionname = event.eventname.cdata
        start = time_to_seconds(event.starttime.cdata)
        duration = time_to_seconds(event.duration.cdata)
        end = start + duration
        start = int(start * 30)
        end = int(end * 30)
        #assert start < 10000 and end < 10000, "Start or end is too big in {}, {}, {}".format(start, end, ann_obj.filename.cdata)

        events.append({
            'actionname': actionname,
            'start': start,
            'end': end
        })
    return events

def print_action_names():
    """
    Print all action names from the data_files.csv xml files to put in the ACTION_LABELS
    """
    def get_action_names(actions):
        action_names = set()
        for action in actions:
            action_names.add(action['actionname'])
        return action_names
    
    df = pd.read_csv(os.path.join(DATA_PATH, 'data_files.csv'))
    actions = set()
    for index, row in df.iterrows():
        xml_file = os.path.join(DATA_PATH, row['path'], row['filename']+'.xml')
        obj = untangle.parse(xml_file)
        xml_actions = get_action_names(parse_actions(obj.annotation))
        actions.update(xml_actions)
    print(actions)


def label_small_frames(actions, num_frames):
    """
    Returns numpy array of shape (num_frames, num_actions)
    Multiple actions can be performed at the same time
    """
    labels = np.zeros((num_frames, len(ACTION_LABELS)), dtype=np.int8)
    for action in actions:
        for i in range(action['start'], action['end']):
            labels[i][ACTION_LABELS.index(action['actionname'])] = 1
    return labels

def label_events(events, num_frames, xml_file_path):
    """
    Returns numpy array of shape (num_frames, num_actions)
    Multiple actions can be performed at the same time
    """
    frames_folder_name = "__".join(xml_file_path.split('/')[5:]).replace('.xml', '_resized')
    num_extracted_frames = len(os.listdir(os.path.join(FRAMES_PATH, frames_folder_name)))
    labels = np.zeros((num_extracted_frames, len(EVENT_LABELS)), dtype=np.int8)
    for event in events:
        for i in range(event['start'], event['end']):
            if i < num_extracted_frames:
                labels[i][EVENT_LABELS.index(event['actionname'])] = 1
            elif i - num_extracted_frames > 10:
                with open(os.path.join("processed_data", "explore_error_files.txt"), 'a') as f:
                    f.write(xml_file_path + '\n')
                print(f"{frames_folder_name}, {num_extracted_frames}, {event['start']}, {event['end']}")
                break
    # select middle of every EVERY_N_FRAMES
    indices = [i + 2 for i in range(0, num_extracted_frames-5, EVERY_N_FRAMES)]
    labels = labels[indices]

    return labels

def parse_xml_for_labels(xml_file_path) -> np.ndarray:
    """
    Parses the xml file and returns a numpy array of shape (num_frames, num_actions)
    Array is multilabel, if action j is performed at frame i, then labels[i][j] = 1

    It will print the number of frames and duration of the video if the number of frames is less than 1000
    If the error happens during the parsing, it will print the filename and the error and continue
    """
    obj = untangle.parse(xml_file_path)
    num_frames = int(obj.annotation.header.frames.cdata)
    # Sometimes the num_frames have negative or 0 values. Calculate the number of frames from the videofile.
    if num_frames < 1000:
        print(num_frames)
        import torchvision
        video_path = xml_file_path[:-4]+'_resized.mp4'
        video_path = video_path.replace('xmls', VIDEO_PATH_NAME)
        reader = torchvision.io.VideoReader(video_path, "video")
        reader_md = reader.get_metadata()
        duration = reader_md['video']['duration'][0]
        if duration == 0:
            raise FileNotFoundError("Video duration is 0 in {}".format(xml_file_path))
        print(duration)
        num_frames = int(duration * 30)
    assert num_frames > 0, "Number of frames must be positive and it was {} in {}".format(num_frames, xml_file_path)
    actions = parse_events(obj.annotation)
    frame_labels = label_events(actions, num_frames, xml_file_path)
    return frame_labels

def create_files_csv():
    """
    Run this first to create data_files.csv on preprocess_data folder
    |filename|path|
    """
    df = pd.DataFrame(columns=['filename', 'path'])
    all_files = list(glob.glob(os.path.join(DATA_PATH, '**', '*.xml'), recursive=True))
    total_files = len(all_files)
    print("There are {} files".format(total_files))
    
    for name in tqdm(all_files, desc="Processing files", ncols=100):
        tqdm.write(f"Currently processing: {name}")
        df = df.append({'filename': os.path.basename(name)[:-4], 'path': os.path.dirname(name)[len(DATA_PATH)+1:]}, ignore_index=True)
        
    df.to_csv(os.path.join("processed_data", "data_files.csv"), index=False)

def create_resized_files_csv():
    """
    Run this first to create data_files.csv on preprocess_data folder
    |filename|path|
    """
    df = pd.DataFrame(columns=['filename', 'path'])
    all_files = list(glob.glob(os.path.join(DATA_PATH, '**', '*.xml'), recursive=True))
    total_files = len(all_files)
    print("There are {} files".format(total_files))
    for name in tqdm(all_files, desc="Processing files", ncols=100):
        tqdm.write(f"Currently processing: {name}")
        
        cur_file_name = os.path.basename(name)[:-4] # remove .xml
        # add _resized.mp4 to the end of the filename
        cur_file_name = cur_file_name + '_resized'
        
        df = df.append({'filename': cur_file_name, 'path': os.path.dirname(name)[len(DATA_PATH)+1:]}, ignore_index=True)
        
    df.to_csv(os.path.join("processed_data", "data_files.csv"), index=False)

def get_dataset():
    """
    Returns a list of tuples (csv_index, filename, labels)
    Code is little bit messy because of the missing xml files 
    I couldn't download the whole dataset because of the size
    """
    dataset = {}
    xmls = glob.glob(os.path.join(DATA_PATH, '**', '*.xml'), recursive=True)
    error_files = set(open(os.path.join("processed_data", "flow_error_files.txt"), 'r').read().split('\n'))
    for xml_file in tqdm(xmls, desc="Processing rows"):
        #xml_file = os.path.join(DATA_PATH, row['path'], row['filename']+'.xml')
        if xml_file in error_files:
            continue
        try:
            labels = parse_xml_for_labels(xml_file)
            resized_folder_name = "__".join(xml_file.split("/")[5:]).replace(".xml", "_resized")
            dataset[resized_folder_name] = labels
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
    return dataset

def get_resized_dataset():
    """
    Returns a list of tuples (csv_index, filename, labels)
    Code is little bit messy because of the missing xml files 
    I couldn't download the whole dataset because of the size
    """
    dataset = []
    df = pd.read_csv(os.path.join("processed_data", 'data_files.csv'))
    total_rows = len(df)
    
    for index, row in tqdm(df.iterrows(), total=total_rows, desc="Processing rows", ncols=100):
        # xml_file = os.path.join(DATA_PATH, row['path'], row['filename']+'.xml')
        filename = row['filename'][:-8]
        
        xml_file = os.path.join(DATA_PATH, row['path'], filename+'.xml')
        
        try:
            labels = parse_xml_for_labels(xml_file)
            dataset.append((index, os.path.join(row['path'], filename + '_resized'), labels))
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
    return dataset

if __name__ == '__main__':
    #create_resized_files_csv()
    dataset = get_dataset()
    print(len(dataset.keys()))
    pickle.dump(dataset, open(os.path.join("processed_data", "dataset_events.pkl"), 'wb'))
# %%
