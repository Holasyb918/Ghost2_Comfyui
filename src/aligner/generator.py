#https://github.com/shrubb/latent-pose-reenactment/tree/master/embedders
import math
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
# import sys
# print(sys.path)
import src.blocks as blocks

class Constant(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.constant = nn.Parameter(torch.ones(1, *shape))

    def forward(self, batch_size):
        return self.constant.expand((batch_size,) + self.constant.shape[1:])


class Generator(nn.Module):
    def __init__(self, padding='zero', in_channels=3, out_channels=3, num_channels=64,
                 max_num_channels=512, d_por=512, d_id=512, d_pose=256, d_exp=256, norm_layer='in',
                 gen_constant_input_size=4, gen_num_residual_blocks=2, output_image_size=512):
        super().__init__()

        def get_res_block(in_channels, out_channels, padding, norm_layer):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=False,
                                   norm_layer=norm_layer)

        def get_up_block(in_channels, out_channels, padding, norm_layer):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=True, downsample=False,
                                   norm_layer=norm_layer)

        if padding == 'zero':
            padding = nn.ZeroPad2d
        elif padding == 'reflection':
            padding = nn.ReflectionPad2d
        else:
            raise Exception('Incorrect `padding` argument, required `zero` or `reflection`')

        assert math.log2(output_image_size / gen_constant_input_size).is_integer(), \
            "`gen_constant_input_size` must be `image_size` divided by a power of 2"
        num_upsample_blocks = int(math.log2(output_image_size / gen_constant_input_size))
        out_channels_block_nonclamped = num_channels * (2 ** num_upsample_blocks)
        out_channels_block = min(out_channels_block_nonclamped, max_num_channels)

        self.constant = Constant(out_channels_block, gen_constant_input_size, gen_constant_input_size)
        current_image_size = gen_constant_input_size

        # Decoder
        layers = []
        for i in range(gen_num_residual_blocks):
            layers.append(get_res_block(out_channels_block, out_channels_block, padding, 'ada' + norm_layer))
        
        for _ in range(num_upsample_blocks):
            in_channels_block = out_channels_block
            out_channels_block_nonclamped //= 2
            out_channels_block = min(out_channels_block_nonclamped, max_num_channels)
            layers.append(get_up_block(in_channels_block, out_channels_block, padding, 'ada' + norm_layer))


        self.decoder_blocks = nn.Sequential(*layers)
        self.rgb_head = nn.Sequential(
            blocks.AdaptiveNorm2d(out_channels_block, norm_layer),
            nn.ReLU(True),
            # padding(1),
            spectral_norm(
                nn.Conv2d(out_channels_block, out_channels, 3, 1, 1),
                eps=1e-4),
            nn.Tanh()
        )
        self.mask_head = nn.Sequential(
            blocks.AdaptiveNorm2d(out_channels_block, norm_layer),
            nn.ReLU(True),
            # padding(1),
            spectral_norm(
                nn.Conv2d(out_channels_block, 1, 3, 1, 1),
                eps=1e-4),
            nn.Sigmoid()
        )

        self.adains = [module for module in self.modules() if module.__class__.__name__ == 'AdaptiveNorm2d']
        self.d_por = d_por
        self.d_id = d_id
        self.d_pose = d_pose
        self.d_exp = d_exp
        
        self.identity_embedding_size = d_por + d_id
        self.pose_embedding_size = d_pose + d_exp

        joint_embedding_size = d_por + d_id + d_pose + d_exp
        self.affine_params_projector = nn.Sequential(
            spectral_norm(nn.Linear(joint_embedding_size, max(joint_embedding_size, 512))),
            nn.ReLU(True),
            spectral_norm(nn.Linear(max(joint_embedding_size, 512), self.get_num_affine_params()))
        )

        self.finetuning = False

    def get_num_affine_params(self):
        return sum(2*module.num_features for module in self.adains)

    def assign_affine_params(self, affine_params):
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveNorm2d":
                new_bias = affine_params[:, :m.num_features]
                new_weight = affine_params[:, m.num_features:2 * m.num_features]

                if m.bias is None: # to keep m.bias being `nn.Parameter`
                    m.bias = new_bias.contiguous()
                else:
                    m.bias.copy_(new_bias)

                if m.weight is None: # to keep m.weight being `nn.Parameter`
                    m.weight = new_weight.contiguous()
                else:
                    m.weight.copy_(new_weight)

                if affine_params.size(1) > 2 * m.num_features:
                    affine_params = affine_params[:, 2 * m.num_features:]

    def assign_embeddings(self, data_dict):
        if self.finetuning:
            identity_embedding = self.identity_embedding.expand(len(data_dict['pose_embed']), -1)
        else:
            identity_embedding = torch.cat((data_dict['por_embed'], data_dict['id_embed']), dim=1) 
            
        pose_embedding = data_dict['pose_embed']
        joint_embedding = torch.cat((identity_embedding, pose_embedding), dim=1)

        affine_params = self.affine_params_projector(joint_embedding)
        self.assign_affine_params(affine_params)

    def enable_finetuning(self, data_dict=None):
        """
            Make the necessary adjustments to generator architecture to allow fine-tuning.
            For `vanilla` generator, initialize AdaIN parameters from `data_dict['embeds']`
            and flag them as trainable parameters.
            Will require re-initializing optimizer, but only after the first call.

            data_dict:
                dict
                Required contents depend on the specific generator. For `vanilla` generator,
                it is `'embeds'` (1 x `args.embed_channels`).
                If `None`, the module's new parameters will be initialized randomly.
        """
        if data_dict is None:
            some_parameter = next(iter(self.parameters())) # to know target device and dtype
            identity_embedding = torch.rand(1, self.identity_embedding_size).to(some_parameter)
        else:
            identity_embedding = torch.cat((data_dict['por_embed'], data_dict['id_embed']), dim=1) # data_dict['embeds']

        if self.finetuning:
            with torch.no_grad():
                self.identity_embedding.copy_(identity_embedding)
        else:
            self.identity_embedding = nn.Parameter(identity_embedding)
            self.finetuning = True

    def forward(self, data_dict):
        self.assign_embeddings(data_dict)

        batch_size = len(data_dict['pose_embed'])
        outputs = self.decoder_blocks(self.constant(batch_size))
        rgbs = self.rgb_head(outputs)
        masks = self.mask_head(outputs)
        
        return {
            'fake_rgbs': rgbs,
            'fake_segm': masks
        }