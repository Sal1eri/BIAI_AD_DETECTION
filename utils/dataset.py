from torch.utils.data import Dataset, DataLoader
import torch
import os
import pandas as pd
from PIL import Image
import yaml
import glob
import nibabel as nib
import numpy as np
from torchvision import transforms
import cv2

import torchvision.utils as vutils

LABEL_MAP = {
    "CN": 0,
    "MCI": 1,
    "EMCI": 2,
    "LMCI": 3,
    "AD": 4
}
def find_nii_path(root_dir, subject_id, image_id):
    """在 Subject 文件夹下递归查找具体的 NIfTI 文件"""
    subject_dir = os.path.join(root_dir, subject_id)
    if not os.path.exists(subject_dir):
        return None
    patterns = [
        os.path.join(subject_dir, "**", f"*{image_id}*.nii"),
        os.path.join(subject_dir, "**", f"*{image_id}*.nii.gz")
    ]
    for pat in patterns:
        files = glob.glob(pat, recursive=True)
        if files:
            return files[0]
    return None

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


class ADNI2DDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, single_slice_mode=False):
        """
        Args:
            csv_file (string): 就是那个 CSV 文件的路径 一般在上个抽象类就传进来了 一般分成train val test】
            root_dir (string): ADNI数据集的根目录 一般在上个抽象类就传进来了
        """
        self.transform = transform
        self.single_slice_mode = single_slice_mode
        self.samples = []
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            group = row['Group']
            if group not in LABEL_MAP:
                continue
            path = find_nii_path(root_dir, row['Subject'], row['Image Data ID'])
            if path:
                self.samples.append({
                    'path': path,
                    'label': LABEL_MAP[group]
                })

    def __len__(self):
        return len(self.samples)
    def _extract_single_slice(self, nii_path):
        img = nib.load(nii_path)
        img = nib.as_closest_canonical(img)
        volume = img.get_fdata(dtype=np.float32)

        z_mid = volume.shape[2] // 2
        axial = volume[:, :, z_mid]

        # Percentile clip
        p1, p99 = np.percentile(axial, (1, 99))
        axial = np.clip(axial, p1, p99)

        # Resize
        axial_resized = cv2.resize(axial, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Normalize
        axial_resized = (axial_resized - axial_resized.mean()) / (axial_resized.std() + 1e-5)

        # 转成三通道
        axial_resized = np.stack([axial_resized]*3, axis=0)  # (3,H,W)
        return axial_resized.astype(np.float32)
    def _extract_3_slices(self, nii_path):
        img = nib.load(nii_path)
        img = nib.as_closest_canonical(img)

        volume = img.get_fdata(dtype=np.float32)

        x_mid = volume.shape[0] // 2
        y_mid = volume.shape[1] // 2
        z_mid = volume.shape[2] // 2

        sagittal = volume[x_mid, :, :]      # (Y,Z)
        coronal  = volume[:, y_mid, :]      # (X,Z)
        axial    = volume[:, :, z_mid]      # (X,Y)

        # ---- 1. percent clip ----
        all_slices = [axial, coronal, sagittal]
        p1, p99 = np.percentile(np.concatenate([s.flatten() for s in all_slices]), (1, 99))
        slices = [np.clip(s, p1, p99) for s in all_slices]

        # ---- 2. resize 到统一大小 (224×224) ----
        resized = []
        for s in slices:
            s = cv2.resize(s, (224, 224), interpolation=cv2.INTER_LINEAR)
            resized.append(s)

        resized = np.stack(resized, axis=0)  # (3, 224, 224)

        # ---- 3. normalize ----
        resized = (resized - resized.mean()) / (resized.std() + 1e-5)

        return resized.astype(np.float32)

    def __getitem__(self, idx):
        item = self.samples[idx]
        nii_path = item["path"]
        
        if self.single_slice_mode:
            # print("Warning: single_slice_mode is enabled.")
            img = self._extract_single_slice(nii_path)
        else:
            img = self._extract_3_slices(nii_path)
        
        img = torch.from_numpy(img)  # (3,H,W) tensor
        
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(item["label"], dtype=torch.long)

        return img, label


class ADNIDataset:
    def __init__(self,model_name):
        self.model_name = model_name
        cfg_pth = os.path.join("configs", "training", f"{model_name}.yaml")
        self.config = load_config(cfg_pth)
        self._loader_map = {
            "vgg16": self._load_vgg16_data,
        }
    def load_data(self, split='train'):
        return self._loader_map[self.model_name](split)
  
    def _load_vgg16_data(self, split='train'):
        assert split in ['train', 'val', 'test'], "Split must be 'train', 'val', or 'test'"
        csv_path = self.config['data'][f"{split}_csv"]
        root_dir = self.config['data']['data_root_dir']
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
        ])
        dataset = ADNI2DDataset(csv_file=csv_path, root_dir=root_dir, transform=transform)
        return dataset


# -------------------------
# ⭐⭐ 你要的可视化三视图函数 ⭐⭐
# -------------------------
def save_example_view(dataset, idx=0, save_path="view_sample.png"):
    """保存三视图合成图（横向拼接）"""
    img, label = dataset[idx]    # img: (3,224,224)

    # (3,H,W) → (1,3,H,W)
    img = img.unsqueeze(0)

    vutils.save_image(img, save_path, normalize=True)
    print(f"Saved: {save_path}, label={label}")


def save_each_slice(dataset, idx=0, out_dir="slices_view"):
    """分别保存 Axial / Coronal / Sagittal 三张图"""
    os.makedirs(out_dir, exist_ok=True)

    img, _ = dataset[idx]    # img: (3,224,224)
    names = ["axial", "coronal", "sagittal"]

    for i in range(3):
        out = os.path.join(out_dir, f"{names[i]}.png")
        vutils.save_image(img[i].unsqueeze(0), out, normalize=True)
        print(f"Saved: {out}")

if __name__ == "__main__":
    dataset = ADNIDataset("vgg16").load_data("val")

    save_example_view(dataset, 0, "view_sample.png")
    save_each_slice(dataset, 0, "slice_views")