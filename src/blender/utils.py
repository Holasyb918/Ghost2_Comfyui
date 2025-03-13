import numpy as np
import torchvision
import torchvision.transforms.functional as TF
import cv2
import torch
import torch.nn.functional as F
from enum import Enum


def encode_face_segmentation(segmentation):
    parse = segmentation
    face_part_ids = [2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20]
    face_map = np.zeros([parse.shape[0], parse.shape[1]], dtype=np.uint8)

    for valid_id in face_part_ids:
        valid_index = np.where(parse==valid_id)
        face_map[valid_index] = 255
    face_map = face_map > 0
    
    return face_map

class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'
    
def dilate(x, dilate_kernel=17):
    dtype = x.dtype
    x = torch.tensor(x)[None, None].to(torch.float32)
    return F.max_pool2d(
        x,
        kernel_size=dilate_kernel, stride=1, 
        padding=dilate_kernel//2
    )[0, 0].numpy().astype(dtype)

def make_composed_random_irregular_mask(mask):
    mask = encode_face_segmentation(mask)
    mask1 = make_random_irregular_mask(mask, draw_method=DrawMethod.LINE)
    mask2 = make_random_irregular_mask(mask, draw_method=DrawMethod.CIRCLE)
    mask3 = make_random_irregular_mask(mask, draw_method=DrawMethod.SQUARE)
    result = mask1 | mask2 | mask3
    return result.astype(np.float32)

def make_random_irregular_mask(mask, max_angle=4, max_len=60, max_width=20, min_times=0, max_times=10,
                               draw_method=DrawMethod.LINE):
    shape = mask.shape
    
    mask_dilated = dilate(mask)
    mask_edge = mask_dilated & ~mask
    
    draw_method = DrawMethod(draw_method)

    height, width = shape
    result_mask = np.zeros((height, width), np.float32)
    times = np.random.randint(min_times, max_times + 1)
    
    try:
        mask_edge_idxs = np.stack(np.nonzero(mask_edge)).T.tolist()
        mask_idxs = np.stack(np.nonzero(mask)).T.tolist()

        for i in range(times):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)
            for j in range(3):
                for _ in range(5):
                    angle = 0.01 + np.random.randint(max_angle)
                    if i % 2 == 0:
                        angle = 2 * 3.1415926 - angle
                    length = 10 + np.random.randint(max_len)
                    brush_w = 5 + np.random.randint(max_width)
                    end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
                    end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)
                    if [int(end_y), int(end_x)] in mask_idxs:
                        continue
                    else:
                        break

                if j % 2 == 0:
                    mask_edge_idx = mask_edge_idxs[np.random.choice(len(mask_edge_idxs))]
                    end_y, end_x = mask_edge_idx

                if draw_method == DrawMethod.LINE:
                    cv2.line(result_mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)

                mask_edge_idx = mask_edge_idxs[np.random.choice(len(mask_edge_idxs))]
                start_y, start_x = mask_edge_idx

                if draw_method == DrawMethod.CIRCLE:
                    cv2.circle(result_mask, (start_x, start_y), radius=brush_w, color=1., thickness=-1)
                if draw_method == DrawMethod.SQUARE:
                    radius = brush_w // 2
                    result_mask[start_y - radius:start_y + radius, start_x - radius:start_x + radius] = 1
            start_x, start_y = end_x, end_y
    except Exception as e:
        pass
    finally:
        return result_mask.astype(bool)

    
def make_affine_augmentation(img, params):
    """
    img: [B, C, H, W] torch tensor
    """
    return TF.affine(img, *params, interpolation=torchvision.transforms.InterpolationMode.NEAREST, fill=0)
    
    
def make_portrait_mask(face_parsing):
    return face_parsing.sum(dim=0, keepdim=True).clamp(0, 1)