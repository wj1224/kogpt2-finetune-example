import numpy as np

import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from config import TrainerConfig

def create_dataloader(batch_size, num_workers, shuffle, tokenizer, config, train=True):
    return DataLoader(DialectDataset(tokenizer, config, train=train),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle)

class DialectDataset(Dataset):
    def __init__(self, tokenizer, config, train=True):
        super(DialectDataset, self).__init__()

        self.tokenizer = tokenizer
        self.datas = pd.read_csv(config.train_path if train else config.test_path)
        self.dialect_token = config.dialect
        self.standard_token = config.standard
        self.bos_token = config.bos
        self.eos_token = config.eos
        self.mask_token = config.mask
        self.pad_token = config.pad
        self.max_len = config.max_len

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas.iloc[idx]
        dialect_form = data["dialect"]
        standard_form = data["standard"]

        dialect_toked = self.tokenizer.tokenize(self.dialect_token + dialect_form)
        standard_toked = self.tokenizer.tokenize(self.standard_token + standard_form + self.eos_token)
        dialect_len, standard_len = len(dialect_toked), len(standard_toked)

        while dialect_len + standard_len > self.max_len:
            rem = max(dialect_len, standard_len) % (self.max_len // 2)
            dialect_toked = dialect_toked[:-rem]
            standard_toked = standard_toked[:-rem - 1] + [standard_toked[-1]]
            dialect_len, standard_len = len(dialect_toked), len(standard_toked)
        assert dialect_len + standard_len <= self.max_len
        
        mask = [1] * (dialect_len + standard_len) + [0] * (self.max_len - dialect_len - standard_len)

        token_ids = self.tokenizer.convert_tokens_to_ids(dialect_toked + standard_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        return (torch.LongTensor(token_ids), torch.LongTensor(mask))

if __name__ == "__main__":
    from config import GeneralConfig
    from transformers import PreTrainedTokenizerFast

    config = GeneralConfig()
    temp_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "skt/kogpt2-base-v2",
        bos_token=config.bos,
        eos_token=config.eos,
        unk_token=config.unk,
        pad_token=config.pad,
        mask_token=config.mask
        )
    temp_loader = create_dataloader(batch_size=1, num_workers=0, shuffle=True, tokenizer=temp_tokenizer, config=config, train=False)

    for idx, (token_ids, mask) in enumerate(temp_loader):
        print(f"Index {idx},\ninput_ids: {token_ids},\nattention_mask: {mask}")
        break