import os
from ZeroshotClassification import ZeroshotClassification
import cv2
from video_utils import _frame_from_video

# config
video_path = os.getcwd() + f"/demo/near_kettle.mp4"
model_name = "ViClip"
class_names = [
    "put hand near socket",
    "near knife",
    "burned",
    "put small object to mouth",
    "running",
    "swimming",
    "drowning",
    "falling",
    "eating",
    "smile",
    "choke",
    "bleeding",
]

#
print("Model loading...")
classify = ZeroshotClassification(class_names, model_name=model_name)
print("Read video frames")
video = cv2.VideoCapture(video_path)
frames = [x for x in _frame_from_video(video)]

#
print("Inference...")
probs, action_predict_index = classify.predict(frames)
print(">>>>> Action predict: ", class_names[action_predict_index.item()])
for index, action in enumerate(class_names):
    print(f"{action: <40} {probs[0][index].item()}")
