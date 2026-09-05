import os
from typing import Any
import torch.nn as nn
import torch

from .utils import Wav2Vec2Config









class Wav2Vec2NormConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias
    ) -> None:
        
        super().__init__() 
        self.conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        self.norm_layer = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        
    
    def forward(self, x: torch.Tensor):
        # shape: (batch, channels, length)
        x = self.conv_layer(x)
        x.transpose_(dim0=-1, dim1=-2)
        x = self.norm_layer(x)
        x = self.activation(x)
        return x
    
    
class Wav2Vec2FeatureEncoder(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config
        
        assert len(config.conv_dim) == len(config.conv_kernels) == len(config.conv_strides), "number of convolution layers, kernels, strides didn't match up"
        num_conv_layers = len(config.conv_dim)
        conv_channels = (1,) + tuple(config.conv_dim) # this add initial channel of our audio, our initial channel will be 1
        
        self.conv_layers = nn.ModuleList()
        for conv_idx in range(num_conv_layers):
            self.conv_layers.append(
                Wav2Vec2NormConvLayer(
                    in_channels=conv_channels[conv_idx], # we use this here cuz conv_channels have 1 at 0 index but other don't, so indexing will be coorect
                    out_channels=self.config.conv_dim[conv_idx],
                    kernel_size=self.config.conv_kernels[conv_idx],
                    stride=self.config.conv_strides[conv_idx],
                    bias=config.conv_bias                    
                )
            )
            
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


        
        
        
    
if __name__ == "__main__":
    x = torch.randn(2, 2, 1000)
    conv = Wav2Vec2NormConvLayer(in_channels=2, out_channels=1, kernel_size=100, stride=100, bias=False)
    x = conv(x)
    print(x.shape)