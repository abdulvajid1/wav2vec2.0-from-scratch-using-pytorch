import numpy as np
from dataclasses import dataclass, asdict
from typing import Literal
from typing import Optional
import torch 
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

@dataclass
class Wav2Vec2ForPreTrainingOutput:

    loss: Optional[torch.FloatTensor] = None
    projected_states: torch.FloatTensor = None
    projected_quantized_states: torch.FloatTensor = None
    codevector_perplexity: torch.FloatTensor = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    diversity_loss: Optional[torch.FloatTensor] = None

@dataclass
class Wav2Vec2Config:

    ### FEATURE ENCODER CONVOLUTION CONFIG ###
    conv_dim: tuple = (512, 512, 512, 512, 512, 512, 512)
    conv_strides: tuple = (5, 2, 2, 2, 2, 2, 2)
    conv_kernels: tuple = (10, 3, 3, 3, 3, 2, 2)
    conv_bias: bool = True
    feature_projection_dropout_p: float = 0.0

    ### POSITIONAL CONVOLUTIONAL EMBEDDING ###
    conv_positional_emb_drop_p: float = 0.0
    conv_positional_emb_groups: int = 16
    conv_positional_emb_kernel_size: int = 128

    ### TRANSFORMER CONFIG ###
    num_transformer_layers: int = 12
    num_attention_heads: int = 12
    embedding_dimension: int = 768
    mlp_ratio: int = 4
    mlp_dropout_p: float = 0.0
    attention_dropout_p: float = 0.0
    transformer_encoder_dropout: float = 0.0
    layer_dropout: float = 0.0
    initializer_range: float = 0.02

    ### GUMBEL SOFTMAX CONFIG ###
    num_codevector_groups: int = 2
    num_codevectors_per_group: int = 320
    codevector_dim: int = 256
    pre_quantizer_dropout: float = 0.0

    ### MASKING CONFIG ###
    masking_probability: float = 0.065
    masking_span_length: int = 10 
    minimum_spans: int = 2

    ### LOSS CONFIG ###
    contrastive_logits_temperature: float = 0.1
    diversity_loss_weight: float = 0.1

    ### TRAINING CONFIG ###
    num_negatives: int = 100

    ### LayerNorm Config ###
    layer_norm_eps: float = 1e-5
    
    ### CTC Config ###
    asr_head_dropout_p: float = 0.1
    blank_token_idx: int = 0
    vocab_size: int = 32

    ### Huggingface Interface Config ###
    hf_model_name: str = "facebook/wav2vec2-base"

    ### Pretrain Backbone Config ###
    path_to_pretrained_weights: str = None

    ### Backbone Config ###
    pretrained_backbone: Literal["pretrained", "pretrained_huggingface", "random"] = "pretrained"

    ### Added in to_dict() method so this Config is compatible with Huggingface Trainer!!! ###
    def to_dict(self):
        return asdict(self)

def compute_encoded_length(input_lengths, conv_kernels, conv_strides):
    """
    Compute the encoded lenght of audio wav input to embedding, Helps to understand the origianl embedding length and padded embedding lenght

    """
    if not isinstance(input_lengths, torch.Tensor):
        input_lengths = torch.tensor(input_lengths)

    def _compute_conv_out(length, kernel, stride):
        """Compute single conv out"""
        return torch.floor((length - (kernel-1) - 1) / stride) + 1

    for kernel, stride in zip(conv_kernels, conv_strides):
        input_lengths = _compute_conv_out(input_lengths, kernel, stride)

    return input_lengths


def compute_sub_attention_mask(config, attention_mask):
    batch_size = attention_mask.shape[0]
    raw_lengths = attention_mask.sum(dim=1)
    encoded_lengths = compute_encoded_length(raw_lengths, config.conv_kernels, config.conv_strides)


    sub_attention_mask = torch.zeros(batch_size, max(encoded_lengths))
    for idx, lenght in enumerate(encoded_lengths):
        sub_attention_mask[idx, :lenght] = 1
    return sub_attention_mask



    

if __name__ == "__main__":
    sampling_lenght = [10000, 15000]
    data = [torch.rand(l) for l in sampling_lenght]
    attention_mask = [torch.ones(l) for l in sampling_lenght]
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)
    
    config = Wav2Vec2Config()
    print(compute_encoded_length(sampling_lenght, config.conv_kernel, config.conv_stride))






   


    