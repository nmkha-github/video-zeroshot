import os, sys
import torch
import numpy as np
from video_utils import frame_preprocess, frames2tensor, resize_frames


class ZeroshotClassification:
    def __init__(self, class_names, model_name="ViClip"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.model_name = model_name
        self.model, self.tokenizer = self.get_model_and_tokenizer()
        self.class_name_feature = self.get_text_feature(class_names)

    def get_model_and_tokenizer(self):
        if self.model_name == "ViClip":
            sys.path.append(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs/ViClip")
            )
            from libs.ViClip.ViClip import ViCLIP
            from libs.ViClip.simple_tokenizer import SimpleTokenizer

            tokenizer = SimpleTokenizer()
            ViCLIP_model = ViCLIP(
                tokenizer,
                pretrain=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "saved_model/ViClip-InternVid-10M-FLT.pth",
                ),
            )
            return ViCLIP_model, tokenizer
        elif self.model_name == "S3D":
            sys.path.append(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs/S3D")
            )
            from libs.S3D.s3dg import S3D

            S3D_model = S3D(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "saved_model/s3d_dict.npy",
                ),
                512,
            )
            S3D_model.load_state_dict(
                torch.load(
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "saved_model/s3d_howto100m.pth",
                    )
                )
            )
            return S3D_model.to(self.device), None

        elif self.model_name == "InternVideo2":
            sys.path.append(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "libs/InternVideo2"
                )
            )
            from libs.InternVideo2.demo.config import (
                Config,
                eval_dict_leaf,
            )
            from libs.InternVideo2.demo.utils import setup_internvideo2

            config = Config.from_file(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "libs/InternVideo2/demo/internvideo2_stage2_config.py",
                )
            )
            config = eval_dict_leaf(config)
            model_pth = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "saved_model/internvl_c_13b_224px.pth",
            )
            config["pretrained_path"] = model_pth
            InternVideo2_model, tokenizer = setup_internvideo2(config)
            return InternVideo2_model, tokenizer

    def get_text_feature(self, list_text):
        with torch.no_grad():
            if self.model_name == "ViClip":
                text_feat_d = {}
                for text in list_text:
                    feature = self.model.get_text_features(
                        text, self.tokenizer, text_feat_d
                    )
                    text_feat_d[text] = feature
                return torch.cat([text_feat_d[text] for text in list_text], 0).to(
                    self.device
                )
            if self.model_name == "S3D":
                self.model.to(self.device)
                features_dict = self.model.text_module.to(self.device)(list_text)
                return features_dict["text_embedding"]

    def predict(self, opencv_frames_list):
        with torch.no_grad():
            self.model.eval()
            if self.model_name == "ViClip":
                self.model = self.model.to(self.device)
                frames_tensor = frames2tensor(opencv_frames_list)
                video_feature = self.model.get_vid_features(frames_tensor)
                prob = self.model.get_predict_label(
                    video_feature, self.class_name_feature
                )
                predict_action = torch.argmax(prob)
                return prob, predict_action
            elif self.model_name == "S3D":
                self.model = self.model.to(self.device)
                frames_tensor = torch.from_numpy(
                    np.array([frame_preprocess(x) for x in opencv_frames_list])
                )
                frames_tensor = (
                    resize_frames(frames_tensor.permute(3, 0, 1, 2).unsqueeze(0))
                    .type(torch.FloatTensor)
                    .to(self.device)
                )
                video_feature = self.model(frames_tensor)["video_embedding"]
                prob = (video_feature @ self.class_name_feature.t()).softmax(dim=-1)
                predict_action = torch.argmax(prob)
                return prob, predict_action
            elif self.model_name == "InternVideo2":
                prob = None
                return prob


# class_names = [
#     "put hand near socket",
#     "near knife",
#     "near kettle",
#     "put small object to mouth",
#     "running",
#     "sitting",
#     "swimming",
#     "drowning",
#     "falling",
#     "eating",
#     "smile",
#     "choke",
#     "bleeding",
# ]
# ViClip_classification = ZeroshotClassification(class_names)
# print("ViClip success")

# InternVideo2_classification = ZeroshotClassification(class_names, "InternVideo2")
# print("InternVideo2 success")
