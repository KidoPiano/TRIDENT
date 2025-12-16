# 此脚本用于从jpg格式的WSI中按照给定spot坐标提取patch，并使用Gigapath模型提取特征，最后保存为.h5文件。
# 最终得到的.h5文件包含四个dataset：
# features, coords, spot_ids, true_labels，分别代表特征矩阵、像素坐标、在玻片上的位置(如6x13)、和病理学家标注的真实标签。
# 此脚本用到了Trident库，请确保已正确安装Trident及其依赖项。并将脚本放在Trident仓库目录中运行。

import os
import pandas as pd
from regex import B
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import timm
from torchvision import transforms
from PIL import Image
import h5py
from typing import Tuple, Any

# 引入 Trident 的组件
from trident.wsi_objects.WSIFactory import load_wsi

# --- 定义 Dataset 类 ---
class SpatialSpotDataset(Dataset):
    def __init__(self, tsv_path: str, wsi_path: str, patch_size: int = 256, transform: Any = None):
        """
        自定义 Dataset：根据 TSV 里的空间坐标读取 WSI Patch
        """
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.transform = transform
        
        # 1. 读取坐标文件
        print(f"Loading coordinates and metadata from {tsv_path}...")
        self.df = pd.read_csv(tsv_path, sep='\t')
        
        # 2. 检查必需的列
        required_cols = ['pixel_x', 'pixel_y', 'xxy', 'label']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"TSV file must contain '{col}' column for alignment and evaluation.")
            
        # 3. 初始化 WSI 对象
        print(f"Loading WSI from {wsi_path}...")
        # 【关键修复】为 ImageWSI 传入 mpp 参数，这里假设为 0.5 (20x)。
        # 请根据您的实际图像扫描倍率进行调整！
        self.wsi_obj = load_wsi(wsi_path, mpp=0.5) 
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Tuple[int, int], str, str]:
        row = self.df.iloc[idx]
        
        # 提取 Spot 中心像素坐标
        cx, cy = int(row['pixel_x']), int(row['pixel_y'])
        
        # 提取元数据（用于后续对齐和评估）
        spot_id = str(row['xxy'])
        true_label = str(row['label'])
        
        # 坐标转换：中心点 -> 左上角
        tl_x = cx - (self.patch_size // 2)
        tl_y = cy - (self.patch_size // 2)
        
        # 读取 Region
        try:
            # location=(x, y), level=0, size=(w, h)
            patch = self.wsi_obj.read_region(
                location=(tl_x, tl_y), 
                level=0, 
                size=(self.patch_size, self.patch_size)
            )
            patch = patch.convert('RGB')
            
        except Exception as e:
            # 捕获读取错误，例如坐标超出 WSI 边界
            print(f"Error reading patch at ({cx}, {cy}). Falling back to black image. Error: {e}")
            patch = Image.new('RGB', (self.patch_size, self.patch_size))

        if self.transform:
            patch = self.transform(patch)
            
        # 返回图像、像素中心坐标、Spot ID 和真实标签
        return patch, (cx, cy), spot_id, true_label

# --- 模型加载函数 ---
def load_gigapath_model():
    print("Loading Gigapath model...")
    # 使用 timm 加载 HuggingFace Hub 上的 Gigapath 模型
    try:
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    except Exception as e:
        print("直接加载 Gigapath 失败，请检查 timm 和 huggingface_hub 是否安装正确，以及网络连接。")
        raise e
        
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

# --- 主运行函数 ---
def main():

    ID_list = ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"]  # 样本ID列表

    for sample_id in ID_list:

        print(f"Processing sample: {sample_id}")
        # --- 配置路径和参数 ---
        tsv_path = f"/mnt/net_sda/rst/M2OST/HER2+/aligned_yzy/{sample_id}_aligned.tsv"
        wsi_path = f"/mnt/net_sda/rst/M2OST/HER2+/images/HE/{sample_id}.jpg"  
        output_dir = "/mnt/net_sda/rst/M2OST/HER2+/embedding_yzy"
        patch_size = 256
        batch_size = 32
        
        os.makedirs(output_dir, exist_ok=True)
        
        # --- 数据预处理 (Gigapath 标准) ---
        transform = transforms.Compose([
            transforms.Resize(224), # Gigapath 模型输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # --- 初始化 Dataset 和 DataLoader ---
        dataset = SpatialSpotDataset(tsv_path, wsi_path, patch_size, transform)
        # 保持 num_workers=0 以避免 WSI 对象序列化问题
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0
        ) 
        
        # --- 加载模型 ---
        model = load_gigapath_model()
        
        # --- 提取特征 ---
        all_features = []
        all_coords = []
        all_spot_ids = []
        all_true_labels = []
        
        print("Starting feature extraction...")
        with torch.no_grad():
            # 接收所有返回值
            for batch_imgs, batch_coords, batch_spot_ids, batch_true_labels in tqdm(dataloader):
                if torch.cuda.is_available():
                    batch_imgs = batch_imgs.cuda()
                
                # 提取特征
                features = model(batch_imgs)
                
                all_features.append(features.cpu().numpy())
                
                # 处理坐标 (从 tuple of tensors 转换为 numpy array)
                coords_np = np.stack([c.numpy() for c in batch_coords], axis=1)
                all_coords.append(coords_np)
                
                # 存储元数据
                all_spot_ids.extend(batch_spot_ids)
                all_true_labels.extend(batch_true_labels)
                
        # --- 保存结果 ---
        if len(all_features) > 0:
            final_features = np.concatenate(all_features, axis=0)
            final_coords = np.concatenate(all_coords, axis=0)
            
            # 将 Spot ID (字符串) 和 Label 转换为 numpy 数组
            # h5py.string_dtype 确保字符串能正确存入 H5 文件
            final_spot_ids = np.array(all_spot_ids, dtype=h5py.string_dtype(encoding='utf-8'))
            final_true_labels = np.array(all_true_labels, dtype=h5py.string_dtype(encoding='utf-8'))
            
            print(f"Extraction complete. Feature shape: {final_features.shape}. Coordinates shape: {final_coords.shape}")
            
            # 建议修改文件名以包含元数据信息
            save_path = os.path.join(output_dir, f"{sample_id}_gigapath_features.h5") 
            
            # 写入 H5 文件
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('features', data=final_features)
                f.create_dataset('coords', data=final_coords)
                f.create_dataset('spot_ids', data=final_spot_ids)
                f.create_dataset('true_labels', data=final_true_labels)
                
            print(f"Successfully saved features and metadata to {save_path}")
        else:
            print("No features extracted. Please check your input data and WSI path.")




if __name__ == "__main__":
    main()