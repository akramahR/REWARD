import numpy as np
from preprocessing import preprocess_video_segment
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
import torch

# Define constants
FIXED_FRAMES = 32

class VideoDataGenerator(Sequence):
    def __init__(self, video_paths, labels, start_times, end_times, fps_list, label_encoder, target_size=(224, 224), target_fps=32):
        self.video_paths = video_paths
        self.labels = labels
        self.start_times = start_times
        self.end_times = end_times
        self.fps_list = fps_list  # Store the list of FPS values
        self.label_encoder = label_encoder
        self.target_size = target_size
        self.target_fps = target_fps
        self.indexes = np.arange(len(self.video_paths))

    def __len__(self):
        return len(self.video_paths)  # Return the total number of videos

    def __getitem__(self, index):
        # Get the single video information
        video_path = self.video_paths[index]
        label = self.labels[index]
        start_time = self.start_times[index]
        end_time = self.end_times[index]
        fps = self.fps_list[index]

        # Preprocess video segment
        tensor = preprocess_video_segment(video_path, start_time, end_time, fps, target_size=self.target_size, target_fps=self.target_fps)

        # Create a one-hot encoded label
        categorical = to_categorical(label, num_classes=len(self.label_encoder.classes_))

        return tensor, categorical  # Return the tensor and the corresponding label

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)  # Shuffle the indexes at the end of each epoch
