from simple_lama_inpainting import SimpleLama
from PIL import Image
import numpy as np
import torch

from .preblending import dilate_torch


class LamaInpainter():
    def __init__(self):
        super().__init__()
        self.simple_lama = SimpleLama()
    
    def __call__(
        self,
        gen_h, # [1, 3, 512, 512], head color reference, seems to be bgr. in [-1, 1]?
        I_tb,  # [1, 3, 512, 512], background, seembs bgr. in [-1, 1]?
        M_Ai,   # [1, 1, 512, 512], inpainting mask, float in [0, 1]
        M_Ah,   # [1, 1, 512, 512], inpainting mask, float in [0, 1]
        I_a
    ):
        dtype = gen_h.dtype
        device = gen_h.device
        
        image_tensor = (I_tb + (I_a * M_Ah)).clamp(-1, 1)
        
        dilated_M_Ah = M_Ah + M_Ai
        dilated_M_Ai = dilate_torch(M_Ai)
        dilated_M_Ai = dilated_M_Ai - (1 - dilated_M_Ah)
        dilated_M_Ai = dilated_M_Ai.clamp(0, 1)
        mask_tensor = dilated_M_Ah # dilated_M_Ai
        
        image = image_tensor.detach().cpu().to(torch.float32).numpy()
        image = (image + 1) / 2
        image = (image * 255).astype(np.uint8)
        image = image[0]
        image = image.transpose(1, 2, 0)
        image = image[:, :, ::-1]
        image = Image.fromarray(image)
        
        mask = mask_tensor.detach().cpu().to(torch.float32).numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = mask[0, 0]
        mask = Image.fromarray(mask)
        result = self.simple_lama(image, mask)
        
        gen_i = np.array(result)
        gen_i = gen_i[:, :, ::-1]
        gen_i = gen_i.transpose(2, 0, 1)
        gen_i = gen_i[None, ...]
        gen_i = gen_i / 255
        gen_i = gen_i * 2 - 1
        gen_i = torch.tensor(gen_i, dtype=dtype, device=device)
        gen_i = gen_i * M_Ai
            
        return gen_i
