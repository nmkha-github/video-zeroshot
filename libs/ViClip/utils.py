from simple_tokenizer import SimpleTokenizer as _Tokenizer
from ViClip import ViCLIP
from video_utils import frames2tensor
import torch
import numpy as np
import cv2

clip_candidates = {"viclip": None, "clip": None}


def get_model_and_tokenizer(name="viclip"):
    global clip_candidates
    m = clip_candidates[name]
    if m is None:
        if name == "viclip":
            tokenizer = _Tokenizer()
            vclip = ViCLIP(tokenizer)
            # m = vclip
            m = (vclip, tokenizer)
        else:
            raise Exception("the target clip model is not found.")

    return m


def get_text_feat_dict(texts, clip, tokenizer, text_feat_d={}):
    for t in texts:
        feat = clip.get_text_features(t, tokenizer, text_feat_d)
        text_feat_d[t] = feat
    return text_feat_d


def get_vid_feat(frames, clip):
    return clip.get_vid_features(frames)


def retrieve_text(
    opencv_frames: list, texts, name="viclip", device=torch.device("cuda")
):
    clip, tokenizer = get_model_and_tokenizer(name)
    clip = clip.to(device)
    frames_tensor = frames2tensor(opencv_frames, device=device)
    vid_feat = get_vid_feat(frames_tensor, clip)
    print("video shape: ", vid_feat.shape)

    text_feat_d = {}
    text_feat_d = get_text_feat_dict(texts, clip, tokenizer, text_feat_d)
    text_feats = [text_feat_d[t] for t in texts]
    text_feats_tensor = torch.cat(text_feats, 0)
    print("text shape: ", text_feats_tensor.shape)

    probs = clip.get_predict_label(vid_feat, text_feats_tensor)

    return probs


def batch_predict(batch_frames, texts, name="viclip", device=torch.device("cuda")):
    clip, tokenizer = get_model_and_tokenizer(name)
    clip = clip.to(device)
    vid_feat = get_vid_feat(batch_frames, clip)

    text_feat_d = {}
    text_feat_d = get_text_feat_dict(texts, clip, tokenizer, text_feat_d)
    text_feats = [text_feat_d[t] for t in texts]
    text_feats_tensor = torch.cat(text_feats, 0)

    probs = clip.get_predict_label(vid_feat, text_feats_tensor)

    return probs


# from video_utils import _frame_from_video

# video = cv2.VideoCapture("ViClip/example1.mp4")
# frames = [x for x in _frame_from_video(video)]

# text_candidates = [
#     "A playful dog and its owner wrestle in the snowy yard, chasing each other with joyous abandon.",
#     "A man in a gray coat walks through the snowy landscape, pulling a sleigh loaded with toys.",
#     "A person dressed in a blue jacket shovels the snow-covered pavement outside their house.",
#     "A pet dog excitedly runs through the snowy yard, chasing a toy thrown by its owner.",
#     "A person stands on the snowy floor, pushing a sled loaded with blankets, preparing for a fun-filled ride.",
#     "A man in a gray hat and coat walks through the snowy yard, carefully navigating around the trees.",
#     "A playful dog slides down a snowy hill, wagging its tail with delight.",
#     "A person in a blue jacket walks their pet on a leash, enjoying a peaceful winter walk among the trees.",
#     "A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.",
#     "A person bundled up in a blanket walks through the snowy landscape, enjoying the serene winter scenery.",
# ]
# frames_tensor = frames2tensor(frames, device=torch.device("cuda"))
# batch_frames = torch.cat((frames_tensor, frames_tensor), 0)
# batch_frames = torch.cat((batch_frames, frames_tensor), 0)
# print(
#     "Batch videos shape (batch, num_frames, channel, width, height):",
#     batch_frames.shape,
# )
# probs = batch_predict(
#     batch_frames, text_candidates, name="viclip", device=torch.device("cuda")
# )
# print(probs.shape)

# probs = retrieve_text(
#     frames, text_candidates, name="viclip", device=torch.device("cuda")
# )
# print(probs)
# for i, prob in enumerate(probs[0]):
#     print(f"text: {text_candidates[i]} ~ prob: {prob}")
