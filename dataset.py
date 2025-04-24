import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, path):
        super().__init__()

        files = os.listdir(path)

        self.files = []

        for file in files:
            self.files.append(os.path.join(path, file))

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        img = Image.open(self.files[idx]).resize((64, 64))
        arr = np.array(img).transpose(2, 0, 1) / 255.

        return (arr - 0.5) * 2.
