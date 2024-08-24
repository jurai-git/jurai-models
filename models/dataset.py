from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.text = self.data.texts
        self.labels = self.data.labels

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        label = str(self.labels[index])

        source = self.tokenizer.batch_encode_plus(
            [text],
            max_length=self.source_len,
            pad_to_max_length=True,
            return_tensors='pt'
        )

        target = self.tokenizer.batch_encode_plus(
            [label],
            max_length=self.target_len,
            pad_to_max_length=True,
            return_tensors='pt'
        )

        source_ids = source['input_ids'].squeeze()
        target_ids = target['input_ids'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'source_mask': source['attention_mask'].squeeze().to(dtype=torch.long),
            'target_mask': target['attention_mask'].squeeze().to(dtype=torch.long)
        }
