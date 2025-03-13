import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

def normalize_and_torch(image: np.ndarray, use_cuda = True) -> torch.tensor:
    """
    Normalize image and transform to torch
    """
    if use_cuda:
        image = torch.tensor(image.copy(), dtype=torch.float32).cuda()
    else:
        image = torch.tensor(image.copy(), dtype=torch.float32)
    if image.max() > 1.:
        image = image/255.
    
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = (image - 0.5) / 0.5

    return image

def get_mask(mask, indexs=[1, 2,3,4,5,6,7,8,9,10,11,12,13, 14,15,16,17,18,19,20]):
    out = torch.zeros_like(mask, device=mask.device)
    for i in indexs:
        out[mask == i] = 1

    return out

def dilate_torch(x, dilate_kernel=17):
    return F.max_pool2d(
        x,
        kernel_size=dilate_kernel, stride=1, 
        padding=dilate_kernel//2
    )

def calc_pseudo_target_bg(target, target_parsing):
    gaussian_blur = transforms.GaussianBlur(3)
    por_mask = get_mask(target_parsing)
    por_mask = dilate_torch(por_mask, dilate_kernel=15)
    inp_mask = dilate_torch(por_mask, dilate_kernel=3) - por_mask
    inp_mask_idxes = inp_mask[0, 0].nonzero()
    por_mask_idxes = por_mask[0, 0].nonzero()
    idxes_dist = (((por_mask_idxes[:, None, :] - inp_mask_idxes[None, :, :]) / 512) ** 2).sum(dim=2)
    min_dist_idx = idxes_dist.min(dim=1)[1]
    inp_mask_colors = gaussian_blur(target)[0, :, inp_mask_idxes[:, 0], inp_mask_idxes[:, 1]].T
    min_dist_value = inp_mask_colors[min_dist_idx, :]
    pseudo_target = target.clone()
    pseudo_target[0, :, por_mask_idxes[:, 0], por_mask_idxes[:, 1]] = min_dist_value.T
    return pseudo_target

def closest_multiple_of_64(n):
    return int(max(64, np.round(n / 64) * 64))

def post_inpainting(result, output, full_frames, M, infer_parsing, pipe):
    x, y = result.shape[1], result.shape[0]
    head_mask = get_mask(infer_parsing(output['fake_rgbs']*output['fake_segm']), indexs=[2,3,4,5,6,7,8,9,10,11,12,14,15,16,17])
    np_source_aligned_mask = ((head_mask[0, 0].detach().cpu().numpy() * 255).astype(np.uint8))
    np_source_aligned_mask_on_full_image = cv2.warpAffine(np_source_aligned_mask, cv2.invertAffineTransform(M), (result.shape[1], result.shape[0]), borderValue=0.0)
        
    hair_mask = get_mask(infer_parsing(normalize_and_torch(full_frames)), indexs=[15,16,17])
    np_target_mask = ((hair_mask[0, 0].detach().cpu().numpy() * 255).astype(np.uint8))    
    
    inpainting_area = np.clip(np_target_mask-np_source_aligned_mask_on_full_image, 0, 255)
    inpainting_area = torch.tensor(inpainting_area.copy(), dtype=torch.float32)/255.
    inpainting_area = inpainting_area.unsqueeze(0).unsqueeze(0).cuda() 
    pil_inp_mask = Image.fromarray((inpainting_area[0, 0].detach().cpu().numpy() * 255).astype(np.uint8))
    
    prompt = "photo of a person" 
    negative_prompt = ""
    generator = torch.Generator("cuda").manual_seed(92)
    image_kand = pipe(prompt=prompt, negative_prompt=negative_prompt, image=Image.fromarray(result), mask_image=pil_inp_mask,height=closest_multiple_of_64(y), width=closest_multiple_of_64(x), generator=generator).images[0]
    image_result = cv2.resize(np.array(image_kand), (x, y))
    return image_result
