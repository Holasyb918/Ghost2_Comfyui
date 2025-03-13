import cv2
import torch
import torchvision
import numpy as np
from skimage import transform as trans

# from repos.emoca.gdl.datasets.ImageDatasetHelpers import bbox2point


def bbox2point(left, right, top, bottom, type='bbox'):
    ''' bbox from detector and landmarks are different
    '''
    if type == 'kpt68':
        old_size = (right - left + bottom - top) / 2 * 1.1
        center_x = right - (right - left) / 2.0
        center_y =  bottom - (bottom - top) / 2.0
        # center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    elif type == 'bbox':
        old_size = (right - left + bottom - top) / 2
        center_x = right - (right - left) / 2.0 
        center_y = bottom - (bottom - top) / 2.0 + old_size * 0.12
        # center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
    elif type == "mediapipe":
        old_size = (right - left + bottom - top) / 2 * 1.1
        center_x = right - (right - left) / 2.0 
        center_y = bottom - (bottom - top) / 2.0
        # center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    else:
        raise NotImplementedError(f" bbox2point not implemented for {type} ")
    if isinstance(center_x, np.ndarray):
        center = np.stack([center_x, center_y], axis=1)
    else: 
        center = np.array([center_x, center_y])
    return old_size, center



src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
#<--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

#---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

#-->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

#-->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])

middle = src3[2, :][None, :]
wide_src = (src - middle) * 0.5 + middle
src_map = {112: src, 224: src * 2}
wide_src_map = {512: wide_src * (512 / 112)}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)


# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode='arcface', wide=False):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        assert image_size == 112
        src = arcface_src
    else:
        cur_src_map = wide_src_map if wide else src_map
        src = cur_src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

def crop_face(image_full: np.ndarray, kps, crop_size=112) -> np.ndarray:
    """
    Crop face from image and resize
    """
    M, _ = estimate_norm(kps, crop_size, mode ='arcface')
    align_img = cv2.warpAffine(image_full, M, (crop_size, crop_size), borderValue=0.0)         
    return align_img

def wide_crop_face(image_full: np.ndarray, kps, crop_size=512, return_M=False):
    """
    Crop face from image and resize
    """
    M, _ = estimate_norm(kps, crop_size, mode ='None', wide=True)
    align_img = cv2.warpAffine(image_full, M, (crop_size, crop_size), borderValue=0.0)
    
    if return_M:
        return align_img, M
    
    return align_img

def emoca_crop(imgs, kpts):
    scale = 1.25
    
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.cpu().numpy()

    if not isinstance(imgs, torch.Tensor):
        imgs = torch.tensor(imgs).permute(0, 3, 1, 2)
        
    left = np.min(kpts[:, :, 0], axis=1)
    right = np.max(kpts[:, :, 0], axis=1)
    top = np.min(kpts[:, :, 1], axis=1)
    bottom = np.max(kpts[:, :, 1], axis=1)
    old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
    center = center.astype(np.int16)
    
    size = (old_size * scale).astype(np.int16)
    half = size // 2
    new_left = (center[:, 0] - half)
    new_right = (center[:, 0] + half)
    new_top = (center[:, 1] - half)
    new_bottom = (center[:, 1] + half)

    crop_list = []
    
    for i in range(imgs.size(0)):
        try:
            crop = torchvision.transforms.functional.crop(imgs[i], new_top[i], new_left[i], (new_bottom[i] - new_top[i]), (new_right[i] - new_left[i]))
            crop = torch.nn.functional.interpolate(crop.unsqueeze(0), size=((224, 224)), mode='bilinear')
        except:
            crop = torch.nn.functional.interpolate(imgs[i].unsqueeze(0), size=((224, 224)), mode='bilinear')
        crop_list.append(crop)
        
    crop_list = torch.cat(crop_list, dim=0)
    return crop_list