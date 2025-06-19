import warnings
import cv2
import matplotlib.pyplot as plt
from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer(
    pose2d="rtmpose-m_8xb64-210e_mpii-256x256",
    device="cuda"
) 

img = cv2.imread("assets\demo_images\3DPW\image_01512.jpg")

result_gen = inferencer(img, show=True)

res = next(result_gen)

persons = res["predictions"][0]

p0             = persons[0]
kpts2d         = p0["keypoints"]
kpts2d_scores  = p0["keypoint_scores"]

print("Detected (x, y):\n", kpts2d)
print("Confidences  :\n", kpts2d_scores)
