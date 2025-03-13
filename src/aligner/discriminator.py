#https://github.com/shrubb/latent-pose-reenactment/tree/master/discriminators

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import src.blocks as blocks

class Discriminator(nn.Module):
    def __init__(self, padding='zero', in_channels=3, out_channels=3, num_channels=64, max_num_channels=512, embed_channels=512,
                 dis_num_blocks=6, image_size=512,
                 num_labels=1):
        super().__init__()

        def get_down_block(in_channels, out_channels, padding):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=True,
                                   norm_layer='none')

        def get_res_block(in_channels, out_channels, padding):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=False,
                                   norm_layer='none')

        if padding == 'zero':
            padding = nn.ZeroPad2d
        elif padding == 'reflection':
            padding = nn.ReflectionPad2d

        self.out_channels = embed_channels

        self.down_block = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels, num_channels, 3, 1, 1),
                eps=1e-4),
            nn.ReLU(),
            spectral_norm(
                nn.Conv2d(num_channels, num_channels, 3, 1, 1),
                eps=1e-4),
            nn.AvgPool2d(2))
        self.skip = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels, num_channels, 1),
                eps=1e-4),
            nn.AvgPool2d(2))

        self.blocks = nn.ModuleList()
        num_down_blocks = min(int(math.log(image_size, 2)) - 2, dis_num_blocks)
        in_channels = num_channels
        for i in range(1, num_down_blocks):
            out_channels = min(in_channels * 2, max_num_channels)
            if i == dis_num_blocks - 1: out_channels = self.out_channels
            self.blocks.append(get_down_block(in_channels, out_channels, padding))
            in_channels = out_channels
        for i in range(num_down_blocks, dis_num_blocks):
            if i == dis_num_blocks - 1: out_channels = self.out_channels
            self.blocks.append(get_res_block(in_channels, out_channels, padding))

        self.linear = spectral_norm(nn.Linear(self.out_channels, 1), eps=1e-4)

        embed = None

        self.finetuning = False

    def pass_inputs(self, input, embed=None):
        scores = []
        feats = []

        out = self.down_block(input)
        out = out + self.skip(input)
        feats.append(out)
        for block in self.blocks:
            out = block(out)
            feats.append(out)
        out = torch.relu(out)
        out = out.view(out.shape[0], self.out_channels, -1).sum(2)
        out_linear = self.linear(out)[:, 0]

        if embed is not None:
            scores.append((out * embed).sum(1) + out_linear)
        else:
            scores.append(out_linear)
        return scores[0], feats

    def enable_finetuning(self, data_dict=None):
        """
            Make the necessary adjustments to discriminator architecture to allow fine-tuning.
            For `vanilla` discriminator, replace embedding matrix W (`self.embed`) with one
            vector `data_dict['embeds']`.

            data_dict:
                dict
                Required contents depend on the specific discriminator. For `vanilla` discriminator,
                it is `'embeds'` (1 x `args.embed_channels`).
        """
        some_parameter = next(iter(self.parameters())) # to know target device and dtype

        if data_dict is None:
            data_dict = {
                'embeds': torch.rand(1, self.out_channels).to(some_parameter)
            }

        with torch.no_grad():
            if self.finetuning:
                self.embed.weight_orig.copy_(data_dict['embeds'])
            else:
                new_embedding_matrix = nn.Embedding(1, self.out_channels).to(some_parameter)
                new_embedding_matrix.weight.copy_(data_dict['embeds'])
                self.embed = spectral_norm(new_embedding_matrix)
                
                self.finetuning = True

    def forward(self, data_dict):
        fake_rgbs = data_dict['fake_rgbs']
        target_rgbs = data_dict['target_rgbs']
        if 'label' in data_dict:
            label = data_dict['label']

        if len(fake_rgbs.shape) > 4:
            fake_rgbs = fake_rgbs[:, 0]
        if len(target_rgbs.shape) > 4:
            target_rgbs = target_rgbs[:, 0]
        
        b, c_in, h, w = target_rgbs.shape

        embed = None
        if hasattr(self, 'embed'):
            embed = self.embed(label)

        fake_in = fake_rgbs
        fake_score_G, fake_features = self.pass_inputs(fake_in, embed)
        fake_score_D, _ = self.pass_inputs(fake_in.detach(), embed.detach() if embed is not None else embed)

        real_in = target_rgbs
        real_score, real_features = self.pass_inputs(real_in, embed)
        
        return {
            'fake_features': fake_features,
            'real_features': real_features,
            'real_embedding': embed,
            'fake_score_G': fake_score_G,
            'fake_score_D': fake_score_D,
            'real_score': real_score
        }