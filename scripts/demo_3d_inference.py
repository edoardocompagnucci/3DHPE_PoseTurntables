import os
import sys
import json
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mmpose.apis import MMPoseInferencer

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.lifter import MLPLifter
from src.utils.transforms import NormalizerJoints2d

# Configuration
MODEL_PATH = r"checkpoints\mlp_lifter_20250519_231621\final_model.pth"  # Replace with your best model path
NUM_JOINTS = 16
IMG_SIZE = 512  # The size used to normalize keypoints
IMAGE_PATH = "assets/demo_03.png"  # Replace with test image path

# Load skeleton structure from bone_graph.json
BONE_GRAPH_PATH = os.path.join(PROJECT_ROOT, "data/meta/bone_graph.json")
with open(BONE_GRAPH_PATH, 'r') as f:
    bone_graph = json.load(f)
    joint_names = bone_graph["order"]
    connections = bone_graph["edges"]

print(f"Loaded skeleton with {len(joint_names)} joints and {len(connections)} connections")

# Function to draw skeleton on image
def draw_skeleton(img, keypoints, connections, color=(0, 255, 0), thickness=2):
    for connection in connections:
        start_idx, end_idx = connection
        x1, y1 = keypoints[start_idx]
        x2, y2 = keypoints[end_idx] 
        
        # Skip drawing if any keypoint is missing (0,0)
        if x1 == 0 and y1 == 0 or x2 == 0 and y2 == 0:
            continue
            
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return img

# Init models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Init 2D pose detector
pose_detector = MMPoseInferencer(
    pose2d="rtmpose-m_8xb64-210e_mpii-256x256",
    device=device.type
)

# 2. Load your 3D pose lifter
model = MLPLifter(num_joints=NUM_JOINTS).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()
print(f"Loaded 3D pose model from: {MODEL_PATH}")

# 3. Load and prepare image
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")
    
img_display = image.copy()
img_h, img_w = image.shape[:2]

# 4. Get 2D keypoints with MMPose
print("Detecting 2D keypoints...")
result_generator = pose_detector(image, show=False)
result = next(result_generator)

# Process detection results
persons = result["predictions"][0]
if not persons:
    print("No person detected in the image.")
    exit()

person = persons[0]  # Take first person
keypoints = person["keypoints"]
confidence = person["keypoint_scores"]

# 5. Draw 2D keypoints on image
for i, (x, y) in enumerate(keypoints):
    if confidence[i] > 0.3:  # Only draw high-confidence points
        cv2.circle(img_display, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(img_display, str(i), (int(x), int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

img_display = draw_skeleton(img_display, keypoints, connections)

# 6. Normalize keypoints for MLPLifter
normalizer = NormalizerJoints2d(img_size=IMG_SIZE)
keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
normalized_keypoints = normalizer({"joints_2d": keypoints_tensor})["joints_2d"]

# 7. Run 3D pose estimation
print("Estimating 3D pose...")
with torch.no_grad():
    pose_3d = model(normalized_keypoints.to(device))
    pose_3d = pose_3d.cpu().numpy().reshape(-1, NUM_JOINTS, 3)

# 8. Visualize results
fig = plt.figure(figsize=(18, 6))

# Show 2D detection
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
plt.title("2D Keypoint Detection")
plt.axis('off')

# Show 3D prediction - front view
ax1 = fig.add_subplot(1, 3, 2, projection='3d')
ax1.set_title("3D Pose - Front View")

# Show 3D prediction - side view
ax2 = fig.add_subplot(1, 3, 3, projection='3d')
ax2.set_title("3D Pose - Side View")

# Get 3D points
X = pose_3d[0, :, 0]
Y = pose_3d[0, :, 1]
Z = pose_3d[0, :, 2]

# Plot skeleton - front view
for connection in connections:
    start_idx, end_idx = connection
    if confidence[start_idx] > 0.3 and confidence[end_idx] > 0.3:
        ax1.plot([X[start_idx], X[end_idx]],
                [Y[start_idx], Y[end_idx]],
                [Z[start_idx], Z[end_idx]], 'b-')

# Plot joints - front view
ax1.scatter(X, Y, Z, c='r', marker='o')

# Add joint indices - front view
for i in range(NUM_JOINTS):
    ax1.text(X[i], Y[i], Z[i], str(i), size=8)

# Set view for front view
ax1.view_init(elev=5, azim=-85)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Plot skeleton - side view
for connection in connections:
    start_idx, end_idx = connection
    if confidence[start_idx] > 0.3 and confidence[end_idx] > 0.3:
        ax2.plot([X[start_idx], X[end_idx]],
                [Y[start_idx], Y[end_idx]],
                [Z[start_idx], Z[end_idx]], 'b-')

# Plot joints - side view
ax2.scatter(X, Y, Z, c='r', marker='o')

# Set view for side view
ax2.view_init(elev=5, azim=5)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# Add joint names in a legend
joint_legend = []
for i, name in enumerate(joint_names):
    joint_legend.append(f"{i}: {name}")

plt.figtext(0.01, 0.01, "\n".join(joint_legend), fontsize=8)

plt.tight_layout()
plt.show()

print("Done!")