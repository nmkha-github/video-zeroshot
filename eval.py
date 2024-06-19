from tqdm import tqdm
from ZeroshotClassification import ZeroshotClassification
import pandas as pd
import cv2
import os
import torch

from video_utils import _frame_from_video

class_names = ["baby laying down", "child sitting", "child standing"]
annotation_df = pd.read_csv(
    os.getcwd() + "/data/Short_Videos/annotation/test.csv"
).filter(["video", "action", "danger"])


def evaluate(annotation_csv, model_name):
    annotation_df = pd.read_csv(os.getcwd() + annotation_csv).filter(
        ["video", "action", "danger"]
    )
    classify = ZeroshotClassification(class_names, model_name=model_name)
    action_accuracy = 0
    total = 0
    for index, row in tqdm(annotation_df.iterrows(), total=annotation_df.shape[0]):
        video_index, action_label, danger_label = (
            row["video"],
            row["action"],
            row["danger"],
        )
        if action_label == 4:
            continue
        video_path = (
            os.getcwd() + f"/data/Short_Videos/Video_{video_index}_{action_label}.mp4"
        )
        video = cv2.VideoCapture(video_path)
        frames = [x for x in _frame_from_video(video)]
        if len(frames) < 8:
            continue
        probs, action_predict_index = classify.predict(frames)
        predict = action_predict_index.item()
        if predict == 0 and action_label == 0:
            action_accuracy += 1
        if predict == 0 and action_label == 1:
            action_accuracy += 1
        if predict == 1 and action_label == 2:
            action_accuracy += 1
        if predict == 2 and action_label == 3:
            action_accuracy += 1
        if predict == 3 and action_label == 4:
            action_accuracy += 1
        total += 1

    print("Action accuracy: ", action_accuracy / total)


print("Test set:")
print("S3D:")
evaluate("/data/Short_Videos/annotation/test.csv", "S3D")
print("ViClip:")
evaluate("/data/Short_Videos/annotation/test.csv", "ViClip")

print("Train set:")
print("S3D:")
evaluate("/data/Short_Videos/annotation/train.csv", "S3D")
print("ViClip:")
evaluate("/data/Short_Videos/annotation/train.csv", "ViClip")
