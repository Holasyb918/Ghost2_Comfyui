from .ghost2_model import (
    LoadAlignerModel,
    LoadBlenderModel,
    LoadInpainterModel,
    LoadFaceAnalysisModel,
    LoadStyleMatteModel,
    LoadFaceParsingModel,
)
from .ghost2_pipeline import FaceAnalysisePipeline, AlignPipeline, FaceParsingPipeline, BlenderPipeline

NODE_CLASS_MAPPINGS = {
    "LoadAlignerModel": LoadAlignerModel,
    "LoadBlenderModel": LoadBlenderModel,
    "LoadInpainterModel": LoadInpainterModel,
    "LoadFaceAnalysisModel": LoadFaceAnalysisModel,
    "LoadStyleMatteModel": LoadStyleMatteModel,
    "LoadFaceParsingModel": LoadFaceParsingModel,
    "FaceAnalysisePipeline": FaceAnalysisePipeline,
    "AlignPipeline": AlignPipeline,
    "FaceParsingPipeline": FaceParsingPipeline,
    "BlenderPipeline": BlenderPipeline,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAlignerModel": "LoadAlignerModel Node",
    "LoadBlenderModel": "LoadBlenderModel Node",
    "LoadInpainterModel": "LoadInpainterModel Node",
    "LoadFaceAnalysisModel": "LoadFaceAnalysisModel Node",
    "LoadStyleMatteModel": "LoadStyleMatteModel Node",
    "LoadFaceParsingModel": "LoadFaceParsingModel Node",
    "FaceAnalysisePipeline": "FaceAnalysisePipeline Node",
    "AlignPipeline": "AlignPipeline Node",
    "FaceParsingPipeline": "FaceParsingPipeline Node",
    "BlenderPipeline": "BlenderPipeline Node",
}
