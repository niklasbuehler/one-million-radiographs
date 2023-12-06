from torch.utils.data import Dataset
from torchvision import transforms

# Define normalization transform (this is z-score; ToTensor already scales data to [0, 1])
#normalize_transform = transforms.Normalize(mean=[0.39119628, 0.39119628, 0.39119628], std=[0.23607571, 0.23607571, 0.23607571])

class IRMADataset(Dataset):
    def __init__(self, df, irma_util, image_size):
        super(IRMADataset, self).__init__()
        self.df = df
        self.irma_util = irma_util
        self.image_size = image_size
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.image_size, self.image_size))])

    def __getitem__(self, index):
        image_path = self.df.iloc[index]['Path']
        label = self.df.iloc[index]['Body Region Label']

        # Use the provided utility function to load and transform the image
        image = self.irma_util.load_image(image_path)
        image = self.transforms(image)
        #print(image[0][126][126])

        return image, label

    def __len__(self):
        return len(self.df)
