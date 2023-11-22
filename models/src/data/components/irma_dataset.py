from torch.utils.data import Dataset
from torchvision import transforms
import torch

class IRMADataset(Dataset):
    def __init__(self, df, irma_util):
        super(IRMADataset, self).__init__()
        self.df = df
        self.irma_util = irma_util
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust the size as needed
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image_path = self.df.iloc[index]['Path']
        label = self.df.iloc[index]['Body Region Label']

        # Use the provided utility function to load and transform the image
        image = self.irma_util.load_image(image_path)
        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.df)
