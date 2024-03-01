import cv2
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def process_pic(path, img_size):
    """
        convert png file to tensor with shape [channels, img_size, img_size]
    """
    x = cv2.imread(path)
    height, width = x.shape[:2]
    crop_size = min([height, width])
    height_x = (height - crop_size + 1) // 2
    width_x = (width - crop_size + 1) // 2
    x = x[height_x : height_x + crop_size, width_x : width_x + crop_size]
    if x.shape[:2] != (img_size, img_size): x = cv2.resize(x, (img_size, img_size))
    x = x.astype('float32')
    x = x / 255 * 2 - 1
    x = torch.tensor(x).permute(2, 0, 1)
    return x

def output_pic(path, x):
    x = x.permute(1, 2, 0)
    x = (x + 1) / 2 * 255
    x = x.cpu().numpy()
    x = np.round(x, 0).astype('uint8')
    cv2.imwrite(path, x)

def show_pic(process, save_path, steps=10):
    """
        process is tensor with shape [batch_size, T, img_size, img_size, channels]
    """
    batch_size = process.shape[0]
    T = process.shape[1]
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(batch_size, steps)
    for n_row in range(batch_size):
        for n_col in range(steps):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            t_idx = (T // steps) * n_col if n_col < steps - 1 else T - 1
            img = process[n_row][t_idx]
            f_ax.imshow(np.round(((img + 1) / 2 * 255).cpu().numpy(), 0).astype('uint8'))
            f_ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def get_all_pics(path):
    top_dir = Path(path)
    pics = [pic for pic in top_dir.rglob('*.png')]
    return pics

class CustomDataset(Dataset):
    def __init__(self, img_path, img_size):
        self.images = [process_pic(str(pic), img_size) for pic in get_all_pics(img_path)]
        self.images = torch.stack(self.images, dim=0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return image