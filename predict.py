import os
import torch
import json
from preprocessing import preprocess_video_segment, processFramesToTensor
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.checkpoint as cu
from slowfast.models import build_model
from sklearn.preprocessing import StandardScaler
from videoGenerator import VideoDataGenerator
from UTILS import Identity
import torch.nn as nn

import pickle

# Load config for SlowFast
args = parse_args()
cfg = load_config(args)
cfg = assert_and_infer_cfg(cfg)

# Constants
TARGET_SIZE = (224, 224)
FPS = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to load and preprocess video
def extract_features_from_video(video_path, start_time, end_time):
    # Create a video data generator with only one video for feature extraction
    tensor = preprocess_video_segment(video_path, start_time, end_time)

    # Initialize SlowFast model
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    model.head = Identity()  # Remove the classifier head to get features
    model.eval()
    model.to(DEVICE)

    # Extract features
    with torch.no_grad():
        video_tensor = tensor.to(dtype=torch.float32).to(DEVICE)
        features = model(video_tensor)
        features = features.cpu().numpy()  # Move features to CPU and convert to NumPy
        return features


# Function to load the trained classifier model and make predictions
def predict_from_features(features):
    # Load the trained classifier model (SimpleNN)
    input_size = features.shape[1]  # Feature size
    num_classes = len(label_encoder.classes_)  # Set this to the number of classes you trained on

    # Define the same model architecture used during training
    class SimpleNN(nn.Module):
        def __init__(self, input_size, num_classes):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)  # First fully connected layer
            self.fc2 = nn.Linear(128, 64)  # Second fully connected layer
            self.fc3 = nn.Linear(64, num_classes)  # Output layer

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    class ImprovedNN(nn.Module):
        def __init__(self, input_size, num_classes):
            super(ImprovedNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 256)  # Increased size
            self.bn1 = nn.BatchNorm1d(256)  # Added batch normalization
            self.fc2 = nn.Linear(256, 128)
            self.bn2 = nn.BatchNorm1d(128)  # Added batch normalization
            self.fc3 = nn.Linear(128, num_classes)  # Output layer
            self.dropout = nn.Dropout(0.5)  # Added dropout

        def forward(self, x):
            x = torch.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)  # Dropout layer
            x = torch.relu(self.bn2(self.fc2(x)))
            x = self.fc3(x)
            return x

    # Load model and its weights
    model = ImprovedNN(input_size, num_classes)
    model.load_state_dict(torch.load("simple_nn_model.pth", map_location=DEVICE))
    model.eval()
    model.to(DEVICE)


    features_tensor = torch.tensor(features, dtype=torch.float32).to(DEVICE)

    # Make predictions
    with torch.no_grad():
        outputs = model(features_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()


# Main function to load a video and make predictions
def process_one_video(video_path, start_time, end_time, fps):
    print(f"Extracting features from video: {video_path}")
    features = extract_features_from_video(video_path, start_time, end_time)

    print("Making predictions from extracted features...")
    predictions = predict_from_features(features)

    predicted_label = label_encoder.inverse_transform([predictions[0]])
    print(f"Predicted Class: {predicted_label}")


# Main function to process all videos in test folder and count correct predictions
def process_all_videos(test_dir):
    total_predictions = 0  # Total number of predictions made
    correct_predictions = 0  # Correctly predicted classes

    for class_folder in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_folder)

        if os.path.isdir(class_path):
            # Iterate over all video files in the class folder
            for video_file in os.listdir(class_path):
                if video_file.endswith('.mp4'):
                    video_path = os.path.join(class_path, video_file)
                    metadata_file = video_path.replace('.mp4', '_metadata.json')

                    # Check if metadata file exists
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            start_time = metadata['start_time']
                            end_time = metadata['end_time']
                            clean_text = metadata['clean_text']
                            fps = metadata.get('fps', 32)  # Default FPS is 32 if not in metadata

                        # Check if the actual class (clean_text) exists in the label encoder
                        if clean_text in label_encoder.classes_:
                            # Extract features from the video
                            print(f"Processing video: {video_file}")
                            features = extract_features_from_video(video_path, start_time, end_time)

                            # Make predictions
                            print("Making predictions from extracted features...")
                            predictions = predict_from_features(features)

                            # Convert prediction to label using the label encoder
                            predicted_label = label_encoder.inverse_transform([predictions[0]])[0]
                            print(f"Predicted Class: {predicted_label}")
                            print(f"Actual Class (clean_text): {clean_text}")

                            # Update counters
                            total_predictions += 1
                            if predicted_label == clean_text:
                                correct_predictions += 1

                            print(f"Correct Prediction: {predicted_label == clean_text}\n")
                        else:
                            print(f"Actual Class '{clean_text}' not found in trained labels, skipping video: {video_file}")
                    else:
                        print(f"Metadata file not found for video: {video_file}")

    # Print final accuracy
    print(f"Total Predictions: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions * 100
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("No valid predictions were made.")



if __name__ == "__main__":
    # Load the label encoder
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    # Path to the test directory containing class folders and videos
    test_dir = "C:\\Asl-interpreter\\scripts\\test_videos"

    # Process all videos in the test directory
    process_all_videos(test_dir)

    #main(video_path, start_time, end_time, fps)

