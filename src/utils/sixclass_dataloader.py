import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image
import cn_clip.clip as clip


def _init_fn(worker_id):
    np.random.seed(2024)


def word2input(texts, vocab_file, max_len):
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True,
                             padding='max_length', truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.size())
    for i, token in enumerate(token_ids):
        masks[i] = (token != 0)
    return token_ids, masks


class SixClassDataset(Dataset):
    def __init__(self, data_list, token_ids, masks, labels, clip_texts,
                 root_dir, mae_transform, clip_transform):
        self.data_list = data_list
        self.token_ids = token_ids
        self.masks = masks
        self.labels = labels
        self.clip_texts = clip_texts
        self.root_dir = root_dir
        self.mae_transform = mae_transform
        self.clip_transform = clip_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        ann = self.data_list[index]
        img_path = os.path.join(self.root_dir, ann['Id'] + ".png")

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224))

        mae_image = self.mae_transform(image)
        clip_image = self.clip_transform(image)

        return (self.token_ids[index],
                self.masks[index],
                self.labels[index],
                torch.tensor(0, dtype=torch.long),
                mae_image,
                clip_image,
                self.clip_texts[index])


class bert_data():
    def __init__(self, max_len, batch_size, vocab_file, category_dict=None, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict

    def load_data(self, path, image_root, shuffle):
        data_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data_list.append(json.loads(line))

        contents = [item['content'] for item in data_list]
        labels = torch.tensor([int(item['label']) for item in data_list], dtype=torch.long)

        token_ids, masks = word2input(contents, self.vocab_file, self.max_len)
        clip_texts = clip.tokenize(contents)

        mae_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        clip_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])

        dataset = SixClassDataset(
            data_list=data_list,
            token_ids=token_ids,
            masks=masks,
            labels=labels,
            clip_texts=clip_texts,
            root_dir=image_root,
            mae_transform=mae_transform,
            clip_transform=clip_transform
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=_init_fn
        )
        return dataloader
