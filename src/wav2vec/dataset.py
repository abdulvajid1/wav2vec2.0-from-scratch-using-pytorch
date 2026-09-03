from utils import (
    compute_span_masking,
    sample_negative_indices,
    compute_sub_attention_mask,
    Wav2Vec2Config
)

from transformers import Wav2Vec2CTCTokenizer

from torch.utils.data import Dataset, DataLoader



class LibriSpeechDataset(Dataset):
    def __init__(
        self,
        path_to_data_root,
        include_split=['dev'],
        max_audio_duration=20.0,
        min_audio_duration=2.0,
        sampling_rate=160000,
        num_audio_channels=1,
        truncate_audio=True,
        return_transcript=True,
        hf_model_name="facebook/wav2vec2-base"
    ):

        if isinstance(include_split, str):
            include_split = [include_split]

        self.sampling_rate = sampling_rate
        self.return_transcript = return_transcript
        self.truncate_audio = truncate_audio
        self.audio
        





    def __len__(self):
        pass

    def __getitem__(self):
        pass