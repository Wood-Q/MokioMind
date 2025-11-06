import json

from torch.utils.data import Dataset
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                # 提取每一行内容放到sample
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # 用tokenizer进行编码
        # 超过max_length的截断，不到的填充
        encoding = self.tokenizer(
            str(sample["text"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding.input_ids.squeeze()
        # 忽略padding产生的Y
        loss_mask = input_ids != self.tokenizer.pad_token_id
        # 第一个到倒数第二个token
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        # 第二个到最后一个token
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask
