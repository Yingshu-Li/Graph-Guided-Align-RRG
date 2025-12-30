import os
import random
import json
import re
import numpy as np
from PIL import Image
import albumentations as A
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor
from configs.config import parser
import pandas as pd
import ast
import torch
from tqdm import tqdm



class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)

        self.image_argument_funcs = [
            # A.RandomBrightnessContrast(p=0.1),
            # A.ToGray(p=0.1),
            # A.ColorJitter(p=0.1),
            # A.RandomResizedCrop(180, 180, p=0.1),
            # A.HorizontalFlip(p=0.1),
            # A.ImageCompression(quality_lower=50, quality_upper=80, p=0.1),
            # A.GaussNoise(p=0.1),
            # A.ToSepia(p=0.1),
            # A.FancyPCA(p=0.1),
            # A.RGBShift(p=0.1),
            # A.Sharpen(p=0.1),
            # A.CoarseDropout(p=0.1)
        ]

        self.image_argument_funcs2 = [
            # A.RandomBrightnessContrast(p=1),
            # A.ToGray(p=1),
            # A.ColorJitter(p=1),
            # A.RandomResizedCrop(180, 180, p=1),
            # A.HorizontalFlip(p=1),
            # A.ImageCompression(quality_lower=50, quality_upper=80, p=1),
            # A.GaussNoise(p=1),
            # A.ToSepia(p=1),
            # A.FancyPCA(p=1),
            # A.RGBShift(p=1),
            # A.Sharpen(p=1),
            # A.CoarseDropout(p=0.1)
        ]


    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0] 

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean_report(self, report):
        # clean Iu-xray reports
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # clean MIMIC-CXR reports
        else:
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .' 
        # report = ' '.join(report.split()[:self.args.max_txt_len])
        return report

    def parse(self, features, training=False):
        to_return = {'id': features['id']}
        report = features.get("report", "")
        report = self.clean_report(report)
        to_return['input_text'] = report
        # if self.dataset != "iu_xray":
        labels = features['labels']
        to_return['label'] = torch.FloatTensor(labels)
        # chest x-ray images
        images = []
        for image_path in features['image_path']:
            with Image.open(os.path.join(self.args.base_dir, image_path)) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                if training:
                    # array = self._do_argument(array)
                    pass
                image = self._parse_image(array) #########
                images.append(image)
        to_return["image"] = images
        return to_return

    def transform_with_parse(self, inputs, training=True):
        return self.parse(inputs, training)

    def _do_argument(self, image, always_apply=False):
        if not always_apply:
            img_argument = random.choice(self.image_argument_funcs)
        else:
            img_argument = random.choice(self.image_argument_funcs2)
        try:
            transformed = img_argument(image=image)  # (width, height, 3)
            transformed_image = transformed["image"]  # (width, height, 3)
        except Exception as e:
            transformed_image = image
        return transformed_image


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.meta = json.load(open(args.annotation, 'r'))
        # random select 1000 samples for quick test
        # shuffle the data
        # random.shuffle(self.meta[split])
        self.meta = self.meta[split]#[:1000]
        self.parser = FieldParser(args)
        self.training = True if split == 'train' else False

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index], self.training)


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'test')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset



if __name__ == '__main__':
    args = parser.parse_args()
    train_dataset , dev_dataset, test_dataset= create_datasets(args)
    # print(train_dataset[0])
    # print(dev_dataset[0])
    print(f'train_data length: {len(train_dataset)} \n test_data length: {len(test_dataset)}')
    for i in tqdm(range(len(train_dataset))):
        print(train_dataset[i])
        a = train_dataset[i]

    for i in tqdm(range(len(test_dataset))):
        # print(train_dataset[i])
        a = test_dataset[i]