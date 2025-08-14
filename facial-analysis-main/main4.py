import cv2
import argparse
import warnings
import numpy as np
import torch
import os
import pickle
from torchvision.transforms import ToPILImage, Resize, ToTensor
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
from deepface import DeepFace
from skimage.transform import SimilarityTransform
from typing import List, Optional
from dataclasses import dataclass, field

# Placeholder for SCRFD and Attribute models (replace with actual imports)
from models import SCRFD, Attribute

warnings.filterwarnings("ignore")

# Load FaceNet model (used for recognition)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
EMBEDDINGS_FILE = "embeddings.pkl"

# Example contour points from your input (for testing)
EXAMPLE_CONTOURS = [
    [(505.149811, 221.201797), (506.987122, 313.285919)],  # Nose bridge
    [(404.642029, 232.854431), (408.527283, 231.366623), (413.565796, 229.427856), (421.378296, 226.967682),
     (432.598755, 225.434143), (442.953064, 226.089508), (453.899811, 228.594818), (461.516418, 232.650467),
     (465.069580, 235.600845), (462.170410, 236.316147), (456.233643, 236.891602), (446.363922, 237.966888),
     (435.698914, 238.149323), (424.320740, 237.235168), (416.037720, 236.012115), (409.983459, 234.870300)],  # Left eye
    [(421.662048, 354.520813), (428.103882, 349.694061), (440.847595, 348.048737), (456.549988, 346.295532),
     (480.526489, 346.089294), (503.375702, 349.470459), (525.624634, 347.352783), (547.371155, 349.091980),
     (560.082031, 351.693268), (570.226685, 354.210175), (575.305420, 359.257751)]  # Top of upper lip
]

# utils/helpers.py content
reference_alignment = np.array(
    [[
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ]],
    dtype=np.float32
)

@dataclass
class Face:
    """
    A class to represent a detected face with its attributes.

    Attributes:
        kps (List[float]): Keypoints of the face (e.g., 5 points for eyes, nose, mouth).
        bbox (List[float]): Bounding box coordinates of the face.
        embedding (List[float]): Face embedding for recognition.
        age (Optional[int]): Age of the detected face.
        gender (Optional[int]): Gender of the detected face (1 for Male, 0 for Female).
        name (Optional[str]): Recognized name from the database.
        contours (List[List[float]]): List of contour points for facial features (e.g., nose bridge, eyes).
    """
    kps: List[float] = field(default_factory=list)
    bbox: List[float] = field(default_factory=list)
    embedding: List[float] = field(default_factory=list)
    age: Optional[int] = None
    gender: Optional[int] = None
    name: Optional[str] = None
    contours: List[List[float]] = field(default_factory=list)

    @property
    def sex(self) -> Optional[str]:
        """Returns the gender as 'M' for Male and 'F' for Female."""
        if self.gender is None:
            return None
        return 'M' if self.gender == 1 else 'F'

def image_alignment(image, center, output_size, scale):
    T = SimilarityTransform(
        scale=scale,
        translation=(output_size / 2 - center[0] * scale, output_size / 2 - center[1] * scale)
    )
    M = T.params[0:2]
    cropped = cv2.warpAffine(image, M, (output_size, output_size), borderValue=0.0)
    return cropped, M

def estimate_norm(landmark, image_size=112):
    assert landmark.shape == (5, 2)
    min_matrix = []
    min_index = []
    min_error = float('inf')
    landmark_transform = np.insert(landmark, 2, values=np.ones(5), axis=1)
    transform = SimilarityTransform()
    if image_size == 112:
        alignment = reference_alignment
    else:
        alignment = float(image_size) / 112 * reference_alignment
    for i in np.arange(alignment.shape[0]):
        transform.estimate(landmark, alignment[i])
        matrix = transform.params[0:2, :]
        results = np.dot(matrix, landmark_transform.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - alignment[i]) ** 2, axis=1)))
        if error < min_error:
            min_error = error
            min_matrix = matrix
            min_index = i
    return min_matrix, min_index

def norm_crop_image(image, landmark, image_size=112, mode='arcface'):
    M, pose_index = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)
    return warped

def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

def draw_corners(image, bbox, color=(0, 255, 0), thickness=3, proportion=0.2):
    x1, y1, x2, y2 = map(int, bbox)
    width = x2 - x1
    height = y2 - y1
    corner_length = int(proportion * min(width, height))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
    cv2.line(image, (x1, y1), (x1 + corner_length, y1), color, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + corner_length), color, thickness)
    cv2.line(image, (x2, y1), (x2 - corner_length, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + corner_length), color, thickness)
    cv2.line(image, (x1, y2), (x1, y2 - corner_length), color, thickness)
    cv2.line(image, (x1, y2), (x1 + corner_length, y2), color, thickness)
    cv2.line(image, (x2, y2), (x2, y2 - corner_length), color, thickness)
    cv2.line(image, (x2, y2), (x2 - corner_length, y2), color, thickness)
    return image

def draw_keypoints(image, keypoints, keypoint_radius=3):
    colors = [
        (0, 0, 255),   # Red
        (0, 255, 0),   # Green
        (255, 0, 0),   # Blue
        (0, 255, 255), # Yellow
        (255, 0, 255)  # Magenta
    ]
    for idx, point in enumerate(keypoints):
        point = point.astype(np.int32)
        color = colors[idx % len(colors)]
        cv2.circle(image, tuple(point), keypoint_radius, color, -1)
    return image

def draw_contours(image, contours, feature_names=None):
    """Draw facial feature contours (e.g., nose bridge, eyes, lips) as polylines with labels.
    Note: This function will skip drawing 'Nose Bridge', 'Left Eye', and 'Top of Upper Lip'."""
    if not contours:
        return image
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # Green, Red, Blue, Yellow, Magenta
    # Define feature names, but exclude specific ones from display
    feature_names = feature_names or []
    excluded_features = ["Nose Bridge", "Left Eye", "Top of Upper Lip"]
    for i, feature_points in enumerate(contours):
        if not feature_points:
            continue
        # Only assign a feature name if it's not in excluded_features
        feature_name = feature_names[i % len(feature_names)] if feature_names and feature_names[i % len(feature_names)] not in excluded_features else ""
        color = colors[i % len(colors)]
        points = np.array([(int(x), int(y)) for x, y in feature_points], dtype=np.int32)
        if len(points) > 1:
            cv2.polylines(image, [points], isClosed=False, color=color, thickness=1)
        for point in points:
            cv2.circle(image, tuple(point), 2, color, -1)
        if len(points) > 0 and feature_name:  # Only draw text if feature_name is not empty
            cv2.putText(image, feature_name, tuple(points[0] + np.array([0, -10])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return image

def put_text(frame, text, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    location = (x1, y1 - 10)
    cv2.putText(
        frame,
        text,
        location,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA
    )

def draw_face_info(frame, face):
    """Draws only the bounding box, 5 keypoints, and text for name, gender, age, and emotion."""
    draw_corners(frame, face.bbox)  # Rectangular box
    draw_keypoints(frame, face.kps)  # 5 dots
    put_text(frame, f"{face.name} | {face.sex} {face.age}", face.bbox)  # Name, gender, age
    cv2.putText(frame, f"Emotion: {face.emotion}", (int(face.bbox[0]), int(face.bbox[1]) - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Emotion

def scale_contours(contours, bbox, frame_shape):
    """Scale contour points to match the bounding box coordinates."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    frame_height, frame_width = frame_shape[:2]
    max_x = max(point[0] for contour in contours for point in contour)
    max_y = max(point[1] for contour in contours for point in contour)
    scale_x = width / max_x if max_x > 0 else 1.0
    scale_y = height / max_y if max_y > 0 else 1.0
    scaled_contours = []
    for contour in contours:
        scaled = [(int(x * scale_x + x1), int(y * scale_y + y1)) for x, y in contour]
        scaled_contours.append(scaled)
    return scaled_contours

# Main script content
def load_models(detection_model_path, attribute_model_path):
    """Loads the face detection, age & gender models, and facenet model for recognition."""
    detection_model = SCRFD(model_path=detection_model_path)
    attribute_model = Attribute(model_path=attribute_model_path)
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
    return detection_model, attribute_model, facenet_model

def load_embeddings():
    """Loads stored face embeddings from file."""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_embedding(name, embedding):
    """Saves a face embedding to the database."""
    embeddings_db = load_embeddings()
    embeddings_db[name] = embedding
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings_db, f)
    print(f"Face embedding saved for {name}!")

def register_face(name, image_path, facenet_model, detection_model):
    """Registers a new person's face embedding from an image."""
    frame = cv2.imread(image_path)
    if frame is None:
        print("Failed to load image")
        return
    boxes_list, _ = detection_model.detect(frame)
    if len(boxes_list) == 0:
        print("No face detected in the provided image.")
        return
    x1, y1, x2, y2, _ = map(int, boxes_list[0])
    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        print("Error: Invalid face region.")
        return
    face_resized = Resize((160, 160))(ToPILImage()(face_crop))
    face_tensor = ToTensor()(face_resized).unsqueeze(0)
    with torch.no_grad():
        embedding = facenet_model(face_tensor).squeeze().numpy()
    save_embedding(name, embedding)
    print(f"Successfully registered face for: {name}")

def process_frame(detection_model, attribute_model, frame, embeddings_database):
    """Detects faces, predicts attributes, recognizes faces, detects emotions, and stores contours in embeddings."""
    boxes_list, points_list = detection_model.detect(frame)
    print("points_list:", points_list)  # Debug: Inspect points_list structure
    for boxes, keypoints in zip(boxes_list, points_list):
        *bbox, conf_score = boxes
        x1, y1, x2, y2 = map(int, bbox)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        # Convert the cropped face for FaceNet model (recognition)
        face_resized = Resize((160, 160))(ToPILImage()(face_crop))
        face_tensor = ToTensor()(face_resized).unsqueeze(0)
        with torch.no_grad():
            embedding = facenet_model(face_tensor).squeeze().numpy()
        recognized_name = "Unknown"
        for name, db_embedding in embeddings_database.items():
            if cosine(embedding, db_embedding) < 0.35:
                recognized_name = name
                break
        # Emotion Detection using DeepFace
        try:
            analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
        except Exception as e:
            emotion = "Unknown"
            print("Emotion detection error:", e)
        # Get gender and age
        gender, age = attribute_model.get(frame, bbox)
        # Prepare keypoints and contours
        kps = keypoints[:5] if len(keypoints) >= 5 else keypoints  # First 5 points for standard keypoints
        contours = keypoints[5:] if len(keypoints) > 5 else scale_contours(EXAMPLE_CONTOURS, [x1, y1, x2, y2], frame.shape)
        # Create face object with emotion and contours
        face = Face(kps=kps, bbox=[x1, y1, x2, y2], age=age, gender=gender, name=recognized_name, contours=contours)
        face.emotion = emotion  # Add emotion to Face object
        # Draw detected information on frame (only box, keypoints, and text)
        draw_face_info(frame, face)
    return frame

def inference_image(detection_model, attribute_model, image_path, save_output, embeddings_database):
    """Processes a single image."""
    frame = cv2.imread(image_path)
    if frame is None:
        print("Failed to load image")
        return
    frame = process_frame(detection_model, attribute_model, frame, embeddings_database)
    if save_output:
        cv2.imwrite(save_output, frame)
    cv2.imshow("FaceDetection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def inference_video(detection_model, attribute_model, video_source, save_output, embeddings_database):
    """Processes a video source for face detection and recognition."""
    cap = cv2.VideoCapture(0 if video_source.isdigit() or video_source == '0' else video_source)
    if not cap.isOpened():
        print("Failed to open video source")
        return
    out = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(save_output, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(detection_model, attribute_model, frame, embeddings_database)
        if save_output:
            out.write(frame)
        cv2.imshow("FaceDetection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows()

def run_face_analysis(detection_weights, attribute_weights, input_source, save_output=None):
    """Runs real-time face detection and recognition."""
    detection_model, attribute_model, facenet_model = load_models(detection_weights, attribute_weights)
    embeddings_database = load_embeddings()
    if isinstance(input_source, str) and input_source.lower().endswith(('.jpg', '.png', '.jpeg')):
        inference_image(detection_model, attribute_model, input_source, save_output, embeddings_database)
    else:
        inference_video(detection_model, attribute_model, input_source, save_output, embeddings_database)

def main():
    """Main function to run face detection or register a face."""
    parser = argparse.ArgumentParser(description="Run face detection and recognition")
    parser.add_argument(
        '--detection-weights',
        type=str,
        default="weights/det_10g.onnx",
        help='Path to the detection model weights file'
    )
    parser.add_argument(
        '--attribute-weights',
        type=str,
        default="weights/genderage.onnx",
        help='Path to the attribute model weights file'
    )
    parser.add_argument('--source', type=str, help='Path to input image/video or "0" for webcam.')
    parser.add_argument('--output', type=str, help='Path to save output image/video.')
    parser.add_argument('--register', type=str, help="Enter your name to register your face.")
    parser.add_argument('--register-image', type=str, help="Path to an image for face registration.")
    args = parser.parse_args()
    if args.register and args.register_image:
        detection_model, attribute_model, facenet_model = load_models(args.detection_weights, args.attribute_weights)
        register_face(args.register, args.register_image, facenet_model, detection_model)
    else:
        run_face_analysis(args.detection_weights, args.attribute_weights, args.source, args.output)

if __name__ == "__main__":
    main()