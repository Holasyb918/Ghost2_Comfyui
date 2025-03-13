from .utils.crops import wide_crop_face, norm_crop  # calc_mask, normalize_and_torch
from .utils.preblending import calc_pseudo_target_bg, post_inpainting
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import rgb_to_grayscale
import cv2


def normalize_and_torch(image: np.ndarray, use_cuda=True) -> torch.tensor:
    """
    Normalize image and transform to torch
    """
    if use_cuda:
        image = torch.tensor(image.copy(), dtype=torch.float32).cuda()
    else:
        image = torch.tensor(image.copy(), dtype=torch.float32)
    if image.max() > 1.0:
        image = image / 255.0

    image = image.permute(2, 0, 1).unsqueeze(0)
    image = (image - 0.5) / 0.5

    return image


def calc_mask(segment_model, img):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).permute(2, 0, 1).cuda()
    if img.max() > 1.0:
        img = img / 255.0
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    input_t = normalize(img)
    input_t = input_t.unsqueeze(0).float()
    with torch.no_grad():
        out = segment_model(input_t)
    result = out[0]

    return result[0]


class FaceAnalysisePipeline:
    def __init__(self):
        self.mean = np.array([0.51315393, 0.48064056, 0.46301059])[None, :, None, None]
        self.std = np.array([0.21438347, 0.20799829, 0.20304542])[None, :, None, None]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "face_analysis_model": ("MODEL",),
                "face_segment_model": ("MODEL",),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "TUPLE",
    )
    RETURN_NAMES = (
        "wide_source",
        "arc_source",
        "mask_source",
        "wide_target",
        "arc_target",
        "mask_target",
        "full_frames",
        "array_2x3_output",
    )
    FUNCTION = "face_analysis"
    CATEGORY = "ghost2/FaceAnalysis"  # 你可以根据需要修改分类

    def process_img(self, img, face_analysis_model, face_segment_model, target=False):
        full_frames = img[0].cpu().numpy() * 255
        dets = face_analysis_model.get(full_frames)
        kps = dets[0]["kps"]
        wide = wide_crop_face(full_frames, kps, return_M=target)
        # print("wide.shape", wide.shape)
        if target:
            wide, M = wide
        arc = norm_crop(full_frames, kps)
        # print("arc.shape", arc.shape)
        mask = calc_mask(face_segment_model, wide)
        # print("mask.shape", mask.shape)
        arc = normalize_and_torch(arc)
        # print("arc.shape", arc.shape)
        wide = normalize_and_torch(wide)
        # print("wide.shape", wide.shape)
        if target:
            return wide, arc, mask, full_frames, M
        return wide, arc, mask

    def face_analysis(
        self, face_analysis_model, face_segment_model, source_image, target_image
    ):

        wide_source, arc_source, mask_source = self.process_img(
            source_image, face_analysis_model, face_segment_model
        )
        wide_target, arc_target, mask_target, full_frames, M = self.process_img(
            target_image, face_analysis_model, face_segment_model, target=True
        )
        return (
            wide_source,
            arc_source,
            mask_source,
            wide_target,
            arc_target,
            mask_target,
            full_frames,
            M,
        )


class AlignPipeline:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "wide_source": ("IMAGE",),
                "arc_source": ("IMAGE",),
                "mask_source": ("IMAGE",),
                "wide_target": ("IMAGE",),
                "arc_target": ("IMAGE",),
                "mask_target": ("IMAGE",),
                "aligner": ("MODEL",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("fake_rgbs", "fake_segm")
    FUNCTION = "align_forward"
    CATEGORY = "ghost2/align"  # 你可以根据需要修改分类

    def align_forward(
        self,
        wide_source,
        arc_source,
        mask_source,
        wide_target,
        arc_target,
        mask_target,
        aligner,
    ):
        try:
            wide_source = wide_source.unsqueeze(1)
            arc_source = arc_source.unsqueeze(1)
            source_mask = mask_source.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            target_mask = mask_target.unsqueeze(0).unsqueeze(0)

            X_dict = {
                "source": {
                    "face_arc": arc_source,
                    "face_wide": wide_source * mask_source,
                    "face_wide_mask": mask_source,
                },
                "target": {
                    "face_arc": arc_target,
                    "face_wide": wide_target * mask_target,
                    "face_wide_mask": mask_target,
                },
            }

            with torch.no_grad():
                output = aligner(X_dict)
            fake_rgbs = output["fake_rgbs"]
            fake_segm = output["fake_segm"]
            print("fake_rgbs", fake_rgbs.shape, fake_rgbs.max(), fake_rgbs.min())
            print("fake_segm", fake_segm.shape, fake_segm.max(), fake_segm.min())
            return (fake_rgbs, fake_segm)

        except Exception as e:
            print(f"Error align: {e}")
            return (None,)  # 或者你可以选择抛出异常，取决于你希望如何处理错误


class FaceParsingPipeline:
    def __init__(self):
        self.mean = np.array([0.51315393, 0.48064056, 0.46301059])[None, :, None, None]
        self.std = np.array([0.21438347, 0.20799829, 0.20304542])[None, :, None, None]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_parsing_sess": ("MODEL",),
                "image": ("IMAGE",),
                # "mask": ("IMAGE", None),
            },
            # "optional": { # 使用 "optional" 键来定义可选输入
            #     "mask": (["IMAGE", "NULL"], { # 类型元组：允许 "IMAGE" 或 "NULL" (None)
            #         "optional": True, # 关键属性：标记为可选
            #         "default": "NULL", # 默认值为 NULL (None)
            #         "tooltip": "可选的图像输入，可以连接 IMAGE 或留空 (None)"
            #     }),
            # },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("face_parsing_mask",)
    FUNCTION = "run_face_parsing_ort"
    CATEGORY = "ghost2/face_parsing"  # 你可以根据需要修改分类

    def run_face_parsing_ort(self, face_parsing_sess, image):  # , mask=None):
        input_name = face_parsing_sess.get_inputs()[0].name
        output_names = [output.name for output in face_parsing_sess.get_outputs()]
        try:
            out = face_parsing_sess.run(
                output_names,
                {
                    input_name: (
                        (
                            (image[:, [2, 1, 0], ...] / 2 + 0.5).cpu().detach().numpy()
                            - self.mean
                        )
                        / self.std
                    ).astype(np.float32)
                },
            )[0]
            print("face parsing out", out.shape, out.max(), out.min())
            return (torch.tensor(out, device="cuda", dtype=torch.float32),)

        except Exception as e:
            print(f"Error face parsing: {e}")
            return (None,)  # 或者你可以选择抛出异常，取决于你希望如何处理错误


class BlenderPipeline:
    def __init__(self):
        self.mean = np.array([0.51315393, 0.48064056, 0.46301059])[None, :, None, None]
        self.std = np.array([0.21438347, 0.20799829, 0.20304542])[None, :, None, None]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target_parsing": ("IMAGE",),
                "wide_target": ("IMAGE",),
                "fake_rgbs": ("IMAGE",),
                "fake_segm": ("IMAGE",),
                "full_frames": ("IMAGE",),
                "array_2x3_output": ("TUPLE",),
                "blender_model": ("MODEL",),
                "face_segment_model": ("MODEL",),
                "inpainter_model": ("MODEL",),
                "face_parsing_model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("face_parsing_mask",)
    FUNCTION = "blender_forward"
    CATEGORY = "ghost2/blend"  # 你可以根据需要修改分类

    def run_face_parsing_ort(self, face_parsing_sess, image):
        input_name = face_parsing_sess.get_inputs()[0].name
        output_names = [output.name for output in face_parsing_sess.get_outputs()]
        out = face_parsing_sess.run(
            output_names,
            {
                input_name: (
                    (
                        (image[:, [2, 1, 0], ...] / 2 + 0.5).cpu().detach().numpy()
                        - self.mean
                    )
                    / self.std
                ).astype(np.float32)
            },
        )[0]
        print("face parsing out", out.shape, out.max(), out.min())
        return torch.tensor(out, device="cuda", dtype=torch.float32)

    def copy_head_back(self, s, t, M):
        mask = np.ones_like(s)
        mask_tr = cv2.warpAffine(
            mask,
            cv2.invertAffineTransform(M),
            (t.shape[1], t.shape[0]),
            borderValue=0.0,
        )
        mask_tr = cv2.erode(mask_tr, np.ones((10, 10)))
        mask_tr = cv2.GaussianBlur(mask_tr, (5, 5), 0)

        image_tr = cv2.warpAffine(
            s, cv2.invertAffineTransform(M), (t.shape[1], t.shape[0]), borderValue=0.0
        )
        res = (t * (1 - mask_tr) + image_tr * mask_tr).astype(np.uint8)
        return res

    def blender_forward(
        self,
        blender_model,
        face_segment_model,
        inpainter_model,
        face_parsing_model,
        target_parsing,
        wide_target,
        fake_rgbs,
        fake_segm,
        full_frames,
        array_2x3_output,
    ):
        array_2x3_output = np.array(array_2x3_output)
        pseudo_norm_target = calc_pseudo_target_bg(wide_target, target_parsing)
        soft_mask = calc_mask(
            face_segment_model,
            ((fake_rgbs * fake_segm)[0, [2, 1, 0], :, :] + 1) / 2,
        )[None]
        new_source = fake_rgbs * soft_mask[:, None, ...] + pseudo_norm_target * (
            1 - soft_mask[:, None, ...]
        )

        mask_source = self.run_face_parsing_ort(
            face_parsing_model, fake_rgbs * fake_segm
        )

        blender_input = {
            "face_source": new_source,  # output['fake_rgbs']*output['fake_segm'] + norm_target*(1-output['fake_segm']),# face_source,
            "gray_source": rgb_to_grayscale(
                new_source[0][[2, 1, 0], ...]
            ).unsqueeze(0),
            "face_target": wide_target,
            "mask_source": mask_source,
            "mask_target": target_parsing,
            "mask_source_noise": None,
            "mask_target_noise": None,
            "alpha_source": soft_mask,
        }
        output_b = blender_model(blender_input, inpainter=inpainter_model)

        np_output = np.uint8(
            (
                output_b["oup"][0]
                .detach()
                .cpu()
                .numpy()
                .transpose((1, 2, 0))[:, :, ::-1]
                / 2
                + 0.5
            )
            * 255
        )
        result = self.copy_head_back(
            np_output, full_frames[..., ::-1], array_2x3_output
        )[None, ]
        # result = result[..., ::-1]
        # result = np.flip(result, axis=-1)
        result = torch.tensor(result, device="cuda", dtype=torch.float32) / 255.0
        result = result.flip(-1)
        print('final ', result.shape)

        return (result,)
