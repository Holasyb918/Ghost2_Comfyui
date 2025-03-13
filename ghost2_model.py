from omegaconf import OmegaConf
import torch
import os
import onnxruntime as ort


from .install import WEIGHTS_PATH, CONFIG_PATH

from .src.aligner.aligner import AlignerModule
from .src.blender.blender import BlenderModule
from .src.inpainter.inpainter import LamaInpainter
from .src.stylematte import StyleMatte


from insightface.app import FaceAnalysis


class LoadAlignerModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "loaders/ghost2/aligner"  # 你可以根据需要修改分类

    def load_model(self):
        try:
            # 读取 config
            config_a = os.path.join(CONFIG_PATH, "aligner.yaml")
            ckpt_a = os.path.join(
                WEIGHTS_PATH, "aligner_checkpoints/aligner_1020_gaze_final.ckpt"
            )

            with open(config_a, "r") as stream:
                cfg_a = OmegaConf.load(stream)
            cfg_a.model.embed.id_encoder = os.path.join(
                WEIGHTS_PATH, "backbone50_1.pth"
            )

            aligner = AlignerModule(cfg_a, inference=True)
            # ckpt = torch.load(ckpt_a, map_location="cpu")
            aligner.load_state_dict(torch.load(ckpt_a), strict=False)
            aligner.eval()
            aligner.cuda()

            return (aligner,)
        except Exception as e:
            print(f"Error loading: {e}")
            return (None,)  # 或者你可以选择抛出异常，取决于你希望如何处理错误


class LoadBlenderModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "loaders/ghost2/blender"  # 你可以根据需要修改分类

    def load_model(self):
        try:
            # 读取 config
            ckpt_b = os.path.join(WEIGHTS_PATH, "blender_checkpoints/blender_lama.ckpt")
            config_b = os.path.join(CONFIG_PATH, "blender.yaml")
            with open(config_b, "r") as stream:
                cfg_b = OmegaConf.load(stream)

            blender = BlenderModule(cfg_b)
            blender.load_state_dict(
                torch.load(ckpt_b, map_location="cpu")["state_dict"],
                strict=False,
            )
            blender.eval()
            blender.cuda()

            return (blender,)
        except Exception as e:
            print(f"Error loading: {e}")
            return (None,)  # 或者你可以选择抛出异常，取决于你希望如何处理错误


class LoadInpainterModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "loaders/ghost2/inpainter"  # 你可以根据需要修改分类

    def load_model(self):
        try:
            inpainter = LamaInpainter()

            return (inpainter,)
        except Exception as e:
            print(f"Error loading: {e}")
            return (None,)  # 或者你可以选择抛出异常，取决于你希望如何处理错误


class LoadFaceAnalysisModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "loaders/ghost2/faceanalysis"  # 你可以根据需要修改分类

    def load_model(self):
        try:
            insightface_root = os.path.join(WEIGHTS_PATH, "insightface")
            app = FaceAnalysis(
                providers=["CUDAExecutionProvider"],
                allowed_modules=["detection"],
                root=insightface_root,
            )
            app.prepare(ctx_id=0, det_size=(640, 640))

            return (app,)
        except Exception as e:
            print(f"Error loading: {e}")
            return (None,)  # 或者你可以选择抛出异常，取决于你希望如何处理错误


class LoadStyleMatteModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "loaders/ghost2/stylematte"  # 你可以根据需要修改分类

    def load_model(self):
        try:
            segment_model_ckpt_path = os.path.join(WEIGHTS_PATH, "stylematte_synth.pth")
            hf_name = "facebook/mask2former-swin-tiny-coco-instance"
            Mask2Former_ckpt_path = os.path.join(WEIGHTS_PATH, hf_name)
            if not os.path.exists(Mask2Former_ckpt_path):
                Mask2Former_ckpt_path = hf_name
            segment_model = StyleMatte(Mask2Former_ckpt_path=Mask2Former_ckpt_path)
            segment_model.load_state_dict(
                torch.load(
                    segment_model_ckpt_path,
                    map_location="cpu",
                )
            )
            segment_model = segment_model.cuda()
            segment_model.eval()

            return (segment_model,)
        except Exception as e:
            print(f"Error loading: {e}")
            return (None,)  # 或者你可以选择抛出异常，取决于你希望如何处理错误


class LoadFaceParsingModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "loaders/ghost2/face_parsing"  # 你可以根据需要修改分类

    def load_model(self):
        try:
            providers = [("CUDAExecutionProvider", {})]
            segformer_model_onnx_path = os.path.join(
                WEIGHTS_PATH, "segformer_B5_ce.onnx"
            )
            parsings_session = ort.InferenceSession(
                segformer_model_onnx_path, providers=providers
            )
            # input_name = parsings_session.get_inputs()[0].name
            # output_names = [output.name for output in parsings_session.get_outputs()]

            return (parsings_session,)
        except Exception as e:
            print(f"Error loading: {e}")
            return (None,)  # 或者你可以选择抛出异常，取决于你希望如何处理错误
