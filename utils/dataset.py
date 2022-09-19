import os
from PIL import Image
import numpy as np
import pandas as pd
import torch.utils.data as data

class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None,
                         names=['name', 'label'])

        if phase == 'train':
            self.data = df[df['name'].str.startswith('train')]
        else:
            self.data = df[df['name'].str.startswith('test')]

        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:,
                     'label'].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {phase} samples: {self.sample_counts}')

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class AffectNet(data.Dataset):
    def __init__(self, aff_path, phase, use_cache=True, transform=None):
        self.phase = phase
        self.transform = transform
        self.aff_path = aff_path

        if use_cache:
            cache_path = os.path.join(aff_path, 'affectnet.csv')
            if os.path.exists(cache_path):
                df = pd.read_csv(cache_path)
            else:
                df = self.get_df()
                df.to_csv(cache_path)
        else:
            df = self.get_df()

        self.data = df[df['phase'] == phase]

        self.file_paths = self.data.loc[:, 'img_path'].values
        self.label = self.data.loc[:, 'label'].values

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {phase} samples: {self.sample_counts}')

    def get_df(self):
        train_path = os.path.join(self.aff_path, 'train_set/')
        val_path = os.path.join(self.aff_path, 'val_set/')
        data = []

        for anno in glob.glob(train_path + 'annotations/*_exp.npy'):
            idx = os.path.basename(anno).split('_')[0]
            img_path = os.path.join(train_path, f'images/{idx}.jpg')
            label = int(np.load(anno))
            data.append(['train', img_path, label])

        for anno in glob.glob(val_path + 'annotations/*_exp.npy'):
            idx = os.path.basename(anno).split('_')[0]
            img_path = os.path.join(val_path, f'images/{idx}.jpg')
            label = int(np.load(anno))
            data.append(['val', img_path, label])

        return pd.DataFrame(data=data, columns=['phase', 'img_path', 'label'])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
