import cv2
import numpy as np
import pickle
from models import ImprovedNN
from slowfast.models import build_model
from preprocessing import processFramesToTensor
import torch
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.checkpoint as cu
from UTILS import Identity

args = parse_args()
cfg = load_config(args)
cfg = assert_and_infer_cfg(cfg)

FIXED_FRAMES = 32  # Desired number of frames


def preprocess_video(video_path, video_fps=32, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return np.zeros((FIXED_FRAMES, *target_size, 3))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame interval to get FIXED_FRAMES from the total number of frames
    frame_interval = max(1, total_frames // FIXED_FRAMES)

    frames = []

    for frame_index in range(FIXED_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index * frame_interval)

        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Frame capture failed at frame index {frame_index}")
            break

        # Perform center crop
        height, width, _ = frame.shape
        crop_size = min(height, width)
        crop_y = (height - crop_size) // 2
        crop_x = (width - crop_size) // 2
        frame = frame[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

        # Resize the frame to the target size
        frame = cv2.resize(frame, target_size)
        frames.append(frame)

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

    tensor = processFramesToTensor(frames)

    return tensor



def extract_features_from_tensor(video_tensor):
    """
    Extract features from the model for a given video tensor.

    Parameters:
        video_tensor (torch.Tensor): The input tensor representing the video.
        model (torch.nn.Module): The model used for feature extraction.
        device (str): The device to run the model on (default: "cuda:0").

    Returns:
        np.ndarray: Extracted features as a NumPy array.
    """
    # Ensure the model is in evaluation mode
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    model.head = Identity()
    model.eval()
    model.to(device="cuda:0")

    # Move tensor to the specified device
    video_tensor = video_tensor.to(device="cuda:0", dtype=torch.float32)

    # Get features from the model without updating the gradients
    with torch.no_grad():
        features = model(video_tensor)
        features_np = features.cpu().numpy()  # Move features to CPU and convert to NumPy

    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    return features_tensor


def load_model_and_scaler(model_path, label_encoder_path):
    # Load the label encoder
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    # Load the trained model
    model = ImprovedNN(512, len(label_encoder.classes_))
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    model.to(device="cuda:0")  # Move model to the same device as before (CPU or GPU)

    return model, label_encoder


def predict(model, label_encoder, features):
    """
    Predict the labels for given features using the trained model.

    Parameters:
        model: The trained PyTorch model.
        scaler: The fitted StandardScaler instance.
        label_encoder: The fitted LabelEncoder instance.
        features: NumPy array of shape (n_samples, 512), the input features.

    Returns:
        A list of predicted labels.
    """

    # Convert the new data to a PyTorch tensor
    #X_new_tensor = torch.tensor(X_new, dtype=torch.float32).to(device)

    # Step 3: Make Predictions
    with torch.no_grad():
        new_outputs = model(features.to(device="cuda:0"))
        _, new_predictions = torch.max(new_outputs, 1)  # Get the class indices of the predictions

    # Step 4: Post-process Predictions
    predicted_labels = label_encoder.inverse_transform(new_predictions.cpu().numpy())

    return predicted_labels

'''

# Usage
model, label_encoder = load_model_and_scaler("simple_nn_model.pth", "label_encoder.pkl")

# Example feature data for prediction
new_data = np.load("newData.npy")  # Replace with your new data file
X_new = new_data[:, :512]  # Extract features

predicted_labels = predict(model, label_encoder, X_new)
print(predicted_labels)  # Print or use the predicted labels as needed

'''

tensors = preprocess_video("C:\\Users\\akram\\OneDrive\\Pictures\\Camera Roll\\me.mp4")
features_extracted = extract_features_from_tensor(tensors)
model, label_encoder = load_model_and_scaler("simple_nn_model.pth", "label_encoder.pkl")
predicted_labels = predict(model, label_encoder, features_extracted)
print(predicted_labels)
