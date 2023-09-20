# %%
import glob
import xml.etree.ElementTree as ET
import os
from xml.sax import SAXParseException
from xml.parsers.expat import ExpatError
import untangle
import numpy as np
import pandas as pd

DATA_PATH = '/mnt/d/ABC/Data'
EVERY_N_FRAMES = 5
ACTION_LABELS = ('normal', 'around', 'pushing', 'stop and go', 'pulling', 'kicking', 'throwing', 'piercing', 'punching', 'threaten', 'falldown')


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


def label_frames(actions, num_frames):
    """
    Returns numpy array of shape (num_frames, num_actions)
    Multiple actions can be performed at the same time
    """
    labels = np.zeros((num_frames, len(ACTION_LABELS)), dtype=np.int8)
    for action in actions:
        for i in range(action['start'], action['end']):
            labels[i][ACTION_LABELS.index(action['actionname'])] = 1
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
        video_path = xml_file_path[:-4]+'.mp4'
        reader = torchvision.io.VideoReader(video_path, "video")
        reader_md = reader.get_metadata()
        duration = reader_md['video']['duration'][0]
        if duration == 0:
            raise FileNotFoundError("Video duration is 0 in {}".format(xml_file_path))
        print(duration)
        num_frames = int(duration * 30)
    assert num_frames > 0, "Number of frames must be positive and it was {} in {}".format(num_frames, xml_file_path)
    actions = parse_actions(obj.annotation)
    frame_labels = label_frames(actions, num_frames)
    return frame_labels

def create_files_csv():
    """
    Run this first to create data_files.csv on DATA_PATH
    |filename|path|
    """
    df = pd.DataFrame(columns=['filename', 'path'])
    for name in glob.glob(os.path.join(DATA_PATH, '**', '*.xml'), recursive=True):
        df = df.append({'filename': os.path.basename(name)[:-4], 'path': os.path.dirname(name)[len(DATA_PATH)+1:]}, ignore_index=True)
    df.to_csv(os.path.join(DATA_PATH, 'data_files.csv'), index=False)

def get_dataset():
    """
    Returns a list of tuples (filename, labels)
    Code is little bit messy because of the missing xml files 
    I couldn't download the whole dataset because of the size
    """
    dataset = []
    df = pd.read_csv(os.path.join(DATA_PATH, 'data_files.csv'))
    for index, row in df.iterrows():
        xml_file = os.path.join(DATA_PATH, row['path'], row['filename']+'.xml')
        try:
            labels = parse_xml_for_labels(xml_file)
            dataset.append((os.path.join(row['path'], row['filename']), labels))
        except FileNotFoundError as e:
            print(e)
            df.drop(index, inplace=True)
    return dataset

if __name__ == '__main__':
    dataset = get_dataset()
    print(len(dataset))
    print(dataset[0])