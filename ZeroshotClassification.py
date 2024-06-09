import os, sys


class ZeroshotClassification:
    def __init__(self, class_names, model_name="ViClip"):
        self.class_names = class_names
        self.model, self.tokenizer = ZeroshotClassification.get_model_and_tokenizer(
            model_name
        )

    @staticmethod
    def get_model_and_tokenizer(model_name):
        if model_name == "ViClip":
            sys.path.append(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib/ViClip")
            )
            from libs.ViClip.ViClip import ViCLIP
            from libs.ViClip.simple_tokenizer import SimpleTokenizer

            tokenizer = SimpleTokenizer()
            ViCLIP_model = ViCLIP(tokenizer)
            return ViCLIP_model, tokenizer

        elif model_name == "InternVideo2":
            sys.path.append(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "lib/InternVideo2"
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
                    "lib/InternVideo2/demo/internvideo2_stage2_config.py",
                )
            )
            config = eval_dict_leaf(config)
            model_pth = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "lib/InternVideo2/internvl_c_13b_224px.pth",
            )
            config["pretrained_path"] = model_pth
            InternVideo2_model, tokenizer = setup_internvideo2(config)
            return InternVideo2_model, tokenizer


class_names = ["stand", "sit"]
ViClip_classification = ZeroshotClassification(class_names)
print("ViClip success")
InternVideo2_classification = ZeroshotClassification(class_names, "InternVideo2")
print("InternVideo2 success")
