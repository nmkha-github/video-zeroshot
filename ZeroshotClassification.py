import os, sys
import torch
from video_utils import frames2tensor


class ZeroshotClassification:
    def __init__(self, class_names, model_name="ViClip"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.model_name = model_name
        self.model, self.tokenizer = self.get_model_and_tokenizer()
        self.class_name_feature = self.get_text_feature(
            class_names, self.model, self.tokenizer
        )

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

    def get_text_feature(self, list_text, model, tokenizer):
        text_feat_d = {}
        for text in list_text:
            feature = model.get_text_features(text, tokenizer, text_feat_d)
            text_feat_d[text] = feature
        return torch.cat([text_feat_d[text] for text in list_text], 0).to(self.device)

    def predict(self, opencv_frames):
        if self.model_name == "ViClip":
            model = model.to(self.device)
            frames_tensor = frames2tensor(opencv_frames)
            video_feature = model.get_vid_features(frames_tensor)
            prob = model.get_predict_label(video_feature, self.class_name_feature)
            return prob
        if self.model_name == "InternVideo2":
            prob = None
            return prob


class_names = ["stand", "sit"]
ViClip_classification = ZeroshotClassification(class_names)
print("ViClip success")

# InternVideo2_classification = ZeroshotClassification(class_names, "InternVideo2")
# print("InternVideo2 success")
