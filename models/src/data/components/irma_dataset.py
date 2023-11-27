from torch.utils.data import Dataset
from torchvision import transforms
import torch

from src.models.components.MinMaxNormalize import MinMaxNormalize

# Define normalization transform
normalize_transform = MinMaxNormalize(0, 255)

class IRMADataset(Dataset):
    def __init__(self, df, irma_util, image_size):
        super(IRMADataset, self).__init__()
        self.df = df
        self.irma_util = irma_util
        self.image_size = image_size
        self.transforms = transforms.Compose(
            [transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor(), normalize_transform]
        )

    def __getitem__(self, index):
        image_path = self.df.iloc[index]['Path']
        label = self.df.iloc[index]['Body Region Label']

        # Use the provided utility function to load and transform the image
        image = self.irma_util.load_image(image_path)
        image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.df)
