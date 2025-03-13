import torch
import torchvision
import torch.nn as nn
import numpy as np
import imgaug.augmenters as iaa
from .iresnet import iresnet50

class Embedder(nn.Module):
    def __init__(self, d_por=512, d_id=512, d_pose=256, d_exp=0, id_encoder="./weights/backbone50_1.pth"):
        super().__init__()
        
        self.d_por = d_por
        self.d_id = d_id
        self.d_pose = d_pose
        self.d_exp = d_exp
        
        self.por_encoder = torchvision.models.resnext50_32x4d(weights='DEFAULT')
        self.por_encoder.fc = nn.Linear(2048, d_por)
        
        self.id_encoder = iresnet50(fp16=True)
        self.id_encoder.load_state_dict(torch.load(id_encoder, map_location='cpu'))
        self.id_encoder.eval()
        for p in self.id_encoder.parameters():
            p.requires_grad = False
        
        self.pose_encoder = torchvision.models.mobilenet_v2(weights='DEFAULT')
        self.pose_encoder.classifier[-1] = nn.Linear(1280, d_pose)
        
        self.finetuning = False

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        augs = [
            sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # scale images to 80-120% of their size, individually per axis
            order=[1],  # use  bilinear interpolation (fast)
            mode=["reflect"]
        )),
            sometimes(iaa.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            order=[1],  # use bilinear interpolation (fast)
            mode=["reflect"]
        ))]
        
        self.augs = iaa.Sequential(augs)

    def enable_finetuning(self):
        self.finetuning = True

    def forward(self, X_dict, use_geometric_augmentations=False):
        
        device = X_dict['source']['face_wide'].device
        dtype = X_dict['source']['face_wide'].dtype
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype)[None, :, None, None]
        def normalize_bgr_to_imagenet(x):
            x = x[:, [2, 1, 0], :, :]
            x = (x + 1) / 2
            x = (x - mean) / std
            return x
        
        por_embed = None
        id_embed = None
        if not self.finetuning:
            bs, K, c, h, w = X_dict['source']['face_wide'].shape

            por_embed = self.por_encoder(
                normalize_bgr_to_imagenet(X_dict['source']['face_wide'].reshape(-1, c, h, w))
            ).reshape(bs, K, -1).mean(dim=1)
        
            with torch.no_grad():
                bs, K, c, h, w = X_dict['source']['face_arc'].shape
                id_embed = self.id_encoder(
                    X_dict['source']['face_arc'].reshape(-1, c, h, w)[:, [2, 1, 0], :, :]
                ).reshape(bs, K, -1).mean(dim=1) # bgr -> rgb!
    
        
            if use_geometric_augmentations:
                
                def tensor2image(img):
                    img = img.cpu().permute(1, 2, 0)[:, :, [2, 1, 0]].numpy() * 0.5 + 0.5
                    return (img * 255).astype(np.uint8)
                
                def image2tensor(img):
                    img = (img / 255 - 0.5) / 0.5
                    return torch.FloatTensor(img[:, :, [2, 1, 0]]).permute(2, 0, 1)

                pic_pose_encoder = [tensor2image(img) for img in X_dict['target']['face_wide']]  
                pic_pose_encoder = self.augs(images = pic_pose_encoder)
                pic_pose_encoder = [image2tensor(img) for img in pic_pose_encoder]
                pic_pose_encoder = torch.stack(pic_pose_encoder, dim=0).to(X_dict['target']['face_wide'].device)
            else:
                pic_pose_encoder = X_dict['target']['face_wide']

            pose_embed = self.pose_encoder(pic_pose_encoder)
        
        return {
            'por_embed': por_embed,
            'id_embed': id_embed,
            'pose_embed': pose_embed,
        }