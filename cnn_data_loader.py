import os
from PIL import Image
from torch.utils.data import Dataset
import torch

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


class PictureDatasetLoader(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.class_list = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

        self.image_files = []

        for f in os.listdir(image_dir):
            emotion_folder = os.path.join(image_dir, f)
            for pct in os.listdir(emotion_folder):
                self.image_files.append({f: os.path.join(emotion_folder, pct)})

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        label, img_name = next(iter(self.image_files[index].items()))
        image = self.transform(Image.open(img_name).convert('RGB'))
        mark = torch.zeros(len(self.class_list))
        mark[self.class_list.index(label)] = 1

        return image, mark
