import cv2
import numpy as np
import os
import json
import torch
import numpy as np
import cv2

FIXED_FRAMES = 32


def preprocess_video_segment(video_path, start_time, end_time, video_fps=32, target_size=(224, 224), target_fps=32):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return np.zeros((FIXED_FRAMES, *target_size, 3))

    start_frame = int(start_time * video_fps)
    end_frame = int(end_time * video_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    frame_interval = max(1, int(video_fps / target_fps))  # Calculate the frame sampling rate


    ''' video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    #Find the next available index for the output directory
    index = 0
    while True:
        output_dir = os.path.join("C:\\Asl-interpreter\\scripts\\tdd\\frames", f"{video_base_name}_{index}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            break
        index += 1'''

    while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Frame capture failed at frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}")
            break

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_interval == 0:
            # Perform center crop
            height, width, _ = frame.shape
            crop_size = min(height, width)
            crop_y = (height - crop_size) // 2
            crop_x = (width - crop_size) // 2
            frame = frame[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
            # Resize the frame to the target size
            frame = cv2.resize(frame, target_size)
            frames.append(frame)

            # todo: delete
            # Save the frame if output_dir is specified
            '''if True:
                frame_index = len(frames) - 1
                frame_filename = os.path.join(output_dir, f"frame_{frame_index}.jpg")
                cv2.imwrite(frame_filename, (frame * 255).astype(np.uint8))  # Convert back to 0-255 range
                print(f"Saved frame: {frame_filename}")'''
            # todo: delete till here

    cap.release()

    # Ensure the output has FIXED_FRAMES
    if len(frames) == 0:
        print(f"Warning: No frames collected, returning zeros for {video_path}")
        frames = np.zeros((FIXED_FRAMES, *target_size, 3))
    else:
        frames = np.array(frames)
        if len(frames) < FIXED_FRAMES:
            print(f"Warning: Padding frames from {len(frames)} to {FIXED_FRAMES} for {video_path}")
            padding = np.zeros((FIXED_FRAMES - len(frames), *target_size, 3))
            frames = np.concatenate([frames, padding], axis=0)
        elif len(frames) > FIXED_FRAMES:
            frames = frames[:FIXED_FRAMES]

    return frames


def processFramesToTensor(frames):
    tensor = torch.from_numpy(frames)
    tensor = tensor_normalize(tensor)
    tensor = torch.permute(tensor, (3, 0, 1, 2))
    tensor = torch.reshape(tensor, (1, 1, tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]))
    return tensor




def tensor_normalize(tensor):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def load_data_and_labels(data_dir):
    data = []
    labels = []

    for sign_dir in os.listdir(data_dir):
        sign_path = os.path.join(data_dir, sign_dir)

        for file_name in os.listdir(sign_path):
            if file_name.endswith('_metadata.json'):
                metadata_path = os.path.join(sign_path, file_name)

                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                video_path = metadata['file']
                start_time = metadata['start_time']
                end_time = metadata['end_time']
                label = metadata['clean_text']

                video_segment = preprocess_video_segment(video_path, start_time, end_time)

                data.append(video_segment)
                labels.append(label)

    return np.array(data), np.array(labels)


def load_video_paths_and_labels(data_dir):
    video_paths = []
    start_times = []
    end_times = []
    labels = []
    fps_list = []  # Add a list to store the fps

    for sign_dir in os.listdir(data_dir):
        sign_path = os.path.join(data_dir, sign_dir)

        for file_name in os.listdir(sign_path):
            if file_name.endswith('_metadata.json'):
                metadata_path = os.path.join(sign_path, file_name)

                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                video_paths.append(metadata['file'])
                start_times.append(metadata['start_time'])
                end_times.append(metadata['end_time'])
                labels.append(metadata['clean_text'])
                fps_list.append(metadata['fps'])

    return video_paths, labels, start_times, end_times, fps_list
