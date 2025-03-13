import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator import BlenderGenerator


class BlenderModule(nn.Module):
    def __init__(self, cfg):
        super(BlenderModule, self).__init__()
        self.gen = BlenderGenerator()
        # self.disc = Discriminator(in_channels=5)
        # self.blender_loss = BlenderLoss(
        #     self.disc,
        #     w_perc_vgg=cfg.train_options.weights.w_perc_vgg,
        #     w_rec=cfg.train_options.weights.w_rec,
        #     w_cycle=cfg.train_options.weights.w_cycle,
        #     w_adv=cfg.train_options.weights.w_adv,
        #     w_reg=cfg.train_options.weights.w_reg
        # )
        # self.g_lr = cfg.train_options.optim.g_lr
        # self.d_lr = cfg.train_options.optim.d_lr
        # self.g_clip = cfg.train_options.optim.g_clip
        # self.d_clip = cfg.train_options.optim.d_clip
        # self.betas = (cfg.train_options.optim.beta1, cfg.train_options.optim.beta2)
        # self.automatic_optimization = False
        # self.save_hyperparameters()
        
    def forward(self, batch, old_version=False, copy_source_attrb=False, inpainter=None):
        oup, gen_h, gen_i, M_Ah, I_tb, M_Ai, I_ag = self.gen(
            batch['face_source'], batch['gray_source'], batch['face_target'],
            batch['mask_source'], batch['mask_target'],
            gt=batch['face_target'],
            M_a_noise=batch['mask_source_noise'], M_t_noise=batch['mask_target_noise'],
            cycle=False, train=False,
            return_inputs=True,
            old_version = old_version,
            copy_source_attrb = copy_source_attrb,
            inpainter=inpainter
        )
        
        return {
            'oup': oup,
            'gen_h': gen_h,
            'gen_i': gen_i,
            'M_Ah': M_Ah,
            'I_tb': I_tb,
            'M_Ai': M_Ai,
            'I_ag': I_ag
        }
    
