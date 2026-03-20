import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, PreTrainedTokenizerFast
from torchvision import transforms
from PIL import Image


def _init_fn(worker_id):
    np.random.seed(2024)


def word2input(texts, bert, max_len):
    tokenizer = BertTokenizer.from_pretrained(bert)
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
    def __init__(self, token_ids, masks, labels, categories, clip_input_ids,
                 clip_attention_mask, image_paths, mae_transform,
                 clip_transform, num_domains=9):
        self.token_ids = token_ids
        self.masks = masks
        self.labels = labels
        self.categories = categories
        self.clip_input_ids = clip_input_ids
        self.clip_attention_mask = clip_attention_mask
        self.image_paths = image_paths
        self.mae_transform = mae_transform
        self.clip_transform = clip_transform
        self.num_domains = num_domains

        self.multi_category = torch.zeros(len(labels), num_domains)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            if index < 3:
                print(f"[WARN] Failed to load image: {img_path} -> {e}")
            img = Image.new('RGB', (224, 224), (0, 0, 0))

        mae_img = self.mae_transform(img)
        clip_img = self.clip_transform(img)

        return (
            self.token_ids[index],
            self.masks[index],
            self.labels[index],
            self.categories[index],
            mae_img,
            clip_img,
            self.clip_input_ids[index],
            self.multi_category[index],
            self.clip_attention_mask[index],
        )


class bert_data():
    def __init__(self, max_len, batch_size, bert, category_dict,
                 num_workers=2, root_dir=None, clip_model='openai/clip-vit-base-patch16'):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bert = bert
        self.category_dict = category_dict
        self.root_dir = root_dir

        self.mae_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.clip_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])

        tokenizer_file = os.path.join(clip_model, 'tokenizer.json') if os.path.isdir(clip_model) else None
        if tokenizer_file and os.path.isfile(tokenizer_file):
            self.clip_tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=tokenizer_file,
                bos_token='<|startoftext|>',
                eos_token='<|endoftext|>',
                pad_token='<|endoftext|>',
            )
        else:
            try:
                from transformers import CLIPTokenizer
                self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model)
            except (ImportError, ValueError):
                self.clip_tokenizer = PreTrainedTokenizerFast.from_pretrained(clip_model)

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

        exist_count = sum(1 for p in image_paths if os.path.isfile(p))
        total = len(image_paths)
        print(f"[DataLoader] {path}: {total} samples, {exist_count}/{total} images found")
        if total > 0:
            print(f"[DataLoader] Sample image paths:")
            for p in image_paths[:3]:
                status = "OK" if os.path.isfile(p) else "MISSING"
                print(f"  [{status}] {p}")
        if exist_count == 0:
            print(f"[WARN] No images found! Check --image_root (current: {self.root_dir})")

        token_ids, masks = word2input(contents, self.bert, self.max_len)

        clip_encoded = self.clip_tokenizer(
            contents, padding='max_length', truncation=True,
            max_length=77, return_tensors='pt'
        )
        clip_input_ids = clip_encoded['input_ids']
        clip_attention_mask = clip_encoded['attention_mask']

        dataset = CustomJsonlDataset(
            token_ids=token_ids,
            masks=masks,
            labels=labels,
            categories=categories,
            clip_input_ids=clip_input_ids,
            clip_attention_mask=clip_attention_mask,
            image_paths=image_paths,
            mae_transform=self.mae_transform,
            clip_transform=self.clip_transform,
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
