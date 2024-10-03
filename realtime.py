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


def preprocess_frame(frame, target_size=(224, 224)):


    # Perform center crop
    height, width, _ = frame.shape
    crop_size = min(height, width)
    crop_y = (height - crop_size) // 2
    crop_x = (width - crop_size) // 2
    frame = frame[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

    # Resize the frame to the target size
    frame = cv2.resize(frame, target_size)


    return frame


def extract_features_from_tensor(video_tensor, model):
    """
    Extract features from the model for a given video tensor.

    Parameters:
        video_tensor (torch.Tensor): The input tensor representing the video.
        model (torch.nn.Module): The model used for feature extraction.

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

    # Make Predictions
    with torch.no_grad():
        new_outputs = model(features.to(device="cuda:0"))

        # Get the softmax probabilities
        probabilities = torch.softmax(new_outputs, dim=1)

        # Get the top 2 predictions and their indices
        top2_probs, top2_indices = torch.topk(probabilities, 2, dim=1)

    # Convert the predicted indices to class labels
    top1_label = label_encoder.inverse_transform(top2_indices[:, 0].cpu().numpy())
    top2_label = label_encoder.inverse_transform(top2_indices[:, 1].cpu().numpy())

    # Convert the probabilities to numpy arrays
    top1_confidence = top2_probs[:, 0].cpu().numpy()
    top2_confidence = top2_probs[:, 1].cpu().numpy()

    # Return both labels and their confidence scores
    return top1_label, top1_confidence, top2_label, top2_confidence


def main():
    # Load model and label encoder
    model, label_encoder = load_model_and_scaler("best_nn_model.pth", "label_encoder.pkl")

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Change 0 to your camera index if needed

    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    # Create and resize the window
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', 800, 600)

    # Initialize variables for frame collection
    frames = []
    frame_count = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        frame = preprocess_frame(frame)
        # Append the preprocessed frame to the list
        frames.append(frame)
        frame_count += 1

        # If we have enough frames, process them and display the prediction
        if frame_count >= FIXED_FRAMES:
            # Create a tensor from the collected frames
            frames = np.array(frames)
            frames = frames[:FIXED_FRAMES]
            frames_tensor = processFramesToTensor(frames)

            # Extract features from the tensor
            features_extracted = extract_features_from_tensor(frames_tensor, model)

            # Make predictions
            top1_label, top1_confidence, top2_label, top2_confidence = predict(model, label_encoder, features_extracted)

            print(f"Top 1 Prediction: {top1_label[0]} (Confidence: {top1_confidence[0]:.2f})")
            print(f"Top 2 Prediction: {top2_label[0]} (Confidence: {top2_confidence[0]:.2f})")
            # Display the prediction on the frame
            #cv2.putText(frame, f"Predicted Label: {predicted_labels[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Clear the frames list and frame count
            frames = []
            frame_count = 0

        # Display the frame
        cv2.imshow('Frame', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()