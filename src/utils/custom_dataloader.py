import json
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name


def _init_fn(worker_id):
    np.random.seed(2024)


def word2input(texts, vocab_file, max_len):
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    token_ids = []
    for text in texts:
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True,
                             padding='max_length', truncation=True)
        )
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.size())
    for i, token in enumerate(token_ids):
        masks[i] = (token != 0)
    return token_ids, masks


class CustomJsonlDataset(Dataset):
    def __init__(self, token_ids, masks, labels, categories, clip_texts,
                 image_paths, mae_transform, clip_preprocess, num_domains=9):
        self.token_ids = token_ids
        self.masks = masks
        self.labels = labels
        self.categories = categories
        self.clip_texts = clip_texts
        self.image_paths = image_paths
        self.mae_transform = mae_transform
        self.clip_preprocess = clip_preprocess
        self.num_domains = num_domains

        self.multi_category = torch.zeros(len(labels), num_domains)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (0, 0, 0))

        mae_img = self.mae_transform(img)
        clip_img = self.clip_preprocess(img)

        return (
            self.token_ids[index],
            self.masks[index],
            self.labels[index],
            self.categories[index],
            mae_img,
            clip_img,
            self.clip_texts[index],
            self.multi_category[index],
        )


class bert_data():
    def __init__(self, max_len, batch_size, vocab_file, category_dict,
                 num_workers=2, root_dir=None):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict
        self.root_dir = root_dir

        self.mae_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        device = "cpu"
        _, self.clip_preprocess = load_from_name("ViT-B-16", device=device, download_root='./')

    def load_data(self, path, shuffle):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        contents = [item['content'] for item in data]
        labels = torch.tensor([int(item['label']) for item in data], dtype=torch.long)
        categories = torch.zeros(len(data), dtype=torch.long)

        if self.root_dir is not None:
            image_paths = [os.path.join(self.root_dir, item['Id'] + ".png") for item in data]
        else:
            data_dir = os.path.dirname(path)
            image_paths = [os.path.join(data_dir, item['Id'] + ".png") for item in data]

        token_ids, masks = word2input(contents, self.vocab_file, self.max_len)
        clip_texts = clip.tokenize(contents)

        dataset = CustomJsonlDataset(
            token_ids=token_ids,
            masks=masks,
            labels=labels,
            categories=categories,
            clip_texts=clip_texts,
            image_paths=image_paths,
            mae_transform=self.mae_transform,
            clip_preprocess=self.clip_preprocess,
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
