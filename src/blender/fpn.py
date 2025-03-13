from torch import nn
import torch
import torchvision
from collections import OrderedDict


class FPN(nn.Module):
    def __init__(
        self,
        out_channels=256,
        out_layer=18
    ):
        super(FPN, self).__init__()
        
        vgg19 = torchvision.models.vgg19(weights='IMAGENET1K_V1').features
        encoder_layers = []
        for module in vgg19.children():
            if module.__class__.__name__ == 'MaxPool2d':
                encoder_layers.append(nn.AvgPool2d(kernel_size=module.kernel_size, stride=module.stride, padding=module.padding))
            else:
                encoder_layers.append(module)
        self.encoder = nn.ModuleList(encoder_layers)
        self.layers_to_save = [0, 4, 9, 18, 27, 36]
        self.decoder = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=[64, 64, 128, 256, 512, 512],
            out_channels=out_channels
        )
        self.out_layer=out_layer
        
    def preprocess_input(self, x):
        x = x[:, [2, 1, 0], :, :]
        x = (x + 1) / 2
        
        input_dtype = x.dtype
        input_device = x.device
        
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=input_dtype, device=input_device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=input_dtype, device=input_device)[None, :, None, None]
        x = (x - mean) / std
        
        return x
        
    def forward(self, x):
        x = self.preprocess_input(x)
        encoder_outputs = OrderedDict()
        
        for i, module in enumerate(self.encoder):
            x = module(x)
            if i in self.layers_to_save:
                encoder_outputs[i] = x
        
        decoder_outputs = self.decoder(encoder_outputs)
        outputs = decoder_outputs[self.out_layer]
        return outputs
