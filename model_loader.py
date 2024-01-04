import os
import torch
from fairseq import checkpoint_utils

from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from src.config import Config
from src.vc_infer_pipeline import VC


class ModelLoader:
    def __init__(self):
        self.model_root = "weights"
        self.config = Config()
        self.model_list = [
            d
            for d in os.listdir(self.model_root)
            if os.path.isdir(os.path.join(self.model_root, d))
        ]
        if len(self.model_list) == 0:
            raise ValueError("No model found in `weights` folder")
        self.model_list.sort()

    def load(self, model_name):
        pth_files = [
            os.path.join(self.model_root, model_name, f)
            for f in os.listdir(os.path.join(self.model_root, model_name))
            if f.endswith(".pth")
        ]
        if len(pth_files) == 0:
            raise ValueError(
                f"No pth file found in {self.model_root}/{model_name}"
            )

        pth_path = pth_files[0]
        print(f"Loading {pth_path}")

        cpt = torch.load(pth_path, map_location="cpu")
        tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        if_f0 = cpt.get("f0", 1)
        version = cpt.get("version", "v1")

        if version == "v1":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs256NSFsid(
                    *cpt["config"], is_half=self.config.is_half
                )
            else:
                net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        elif version == "v2":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs768NSFsid(
                    *cpt["config"], is_half=self.config.is_half
                )
            else:
                net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
        else:
            raise ValueError("Unknown version")

        del net_g.enc_q
        net_g.load_state_dict(cpt["weight"], strict=False)
        print("Model loaded")
        net_g.eval().to(self.config.device)

        if self.config.is_half:
            net_g = net_g.half()
        else:
            net_g = net_g.float()

        vc = VC(tgt_sr, self.config)

        index_files = [
            os.path.join(self.model_root, model_name, f)
            for f in os.listdir(os.path.join(self.model_root, model_name))
            if f.endswith(".index")
        ]

        if len(index_files) == 0:
            print("No index file found")
            index_file = ""
        else:
            index_file = index_files[0]
            print(f"Index file found: {index_file}")

        return (
            tgt_sr,
            net_g,
            vc,
            version,
            index_file,
            if_f0,
        )

    def load_hubert(self):
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            ["hubert_base.pt"],
            suffix="",
        )
        self.hubert_model = models[0]
        self.hubert_model = self.hubert_model.to(self.config.device)

        if self.config.is_half:
            self.hubert_model = self.hubert_model.half()
        else:
            self.hubert_model = self.hubert_model.float()

        return self.hubert_model.eval()
