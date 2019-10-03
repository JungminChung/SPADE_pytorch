import torch, os
from PIL import Image
from torch.utils.data.dataset import Dataset

class customDataset(Dataset):
    def __init__(self, origin_path, segmen_path, transform=None):
        self.origin_path = origin_path
        self.segmen_path = segmen_path
        self.transform = transform
        self.name_list = os.listdir(self.origin_path)

    def __getitem__(self, index):
        file_name = self.name_list[index]

        origin_img = Image.open(os.path.join(self.origin_path, file_name))
        segmen_img = Image.open(os.path.join(self.segmen_path, file_name))

        if self.transform : 
            origin_img = self.transform(origin_img)
            segmen_img = self.transform(segmen_img)

        return origin_img, segmen_img

    def __len__(self):
        return len(self.name_list)