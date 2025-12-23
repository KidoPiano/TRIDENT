# 空间聚类的数据集均包含在STimage-1K4M中
# 此脚本专门适配 STimage-1K4M 格式的数据集结构
# 自动遍历多个数据集，提取 patch 并使用 Gigapath 提取特征，保存为 .h5

import os
import pandas as pd
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
    def __init__(self, coord_df: pd.DataFrame, wsi_path: str, patch_size: int = 256, transform: Any = None):
        """
        根据统一后的 CSV 坐标读取图像 Patch
        """
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.transform = transform
        self.df = coord_df
        
        # 加载图像对象 (Trident 会根据后缀名处理 png/jpg)
        # 假设 MPP 为 0.5 (20x)，如果不确定，该参数在处理非 WSI 格式时主要影响尺度计算
        self.wsi_obj = load_wsi(wsi_path, mpp=0.5) 
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Tuple[int, int], str, str]:
        row = self.df.iloc[idx]
        
        # 对应新列名: xaxis -> x, yaxis -> y
        cx, cy = int(row['xaxis']), int(row['yaxis'])
        
        # 提取元数据
        spot_id = str(row['name'])
        true_label = str(row['label'])
        
        # 坐标转换：中心点 -> 左上角
        tl_x = cx - (self.patch_size // 2)
        tl_y = cy - (self.patch_size // 2)
        
        try:
            # Trident 的 read_region 接口
            patch = self.wsi_obj.read_region(
                location=(tl_x, tl_y), 
                level=0, 
                size=(self.patch_size, self.patch_size)
            )
            patch = patch.convert('RGB')
        except Exception as e:
            patch = Image.new('RGB', (self.patch_size, self.patch_size))

        if self.transform:
            patch = self.transform(patch)
            
        return patch, (cx, cy), spot_id, true_label

# --- 模型加载函数 ---
def load_gigapath_model():
    print("Loading Gigapath model...")
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

# --- 主运行函数 ---
def main():

    # 设置各数据集对应的 patch 大小
    dataset_configs = {
        "Maynard_visium": {"patch_size": 128},
        "GSE213688_visium": {"patch_size": 24},
        "Erickson_visium": {"patch_size": 150},
        "Chen_": {"patch_size": 0}, 
        "Andersson_ST": {"patch_size": 256}
    }
    # 默认 patch 大小（如果字典里没找到）
    default_patch_size = 256

    # 根目录配置
    base_dir = "/mnt/net_sda/rst/Sub_dataset_for_spatial_cluster_yzy"
    output_root = "/mnt/net_sda/rst/Sub_dataset_for_spatial_cluster_yzy/embedding_results"
    
    # 数据集列表
    datasets = ["Erickson_visium", "GSE213688_visium"]  #"Maynard_visium", "GSE213688_visium", "Erickson_visium", "Chen_", "Andersson_ST"
    batch_size = 128
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载模型 (全局加载一次即可)
    model = load_gigapath_model()

    for ds_name in datasets:
        # --- 2. 动态调取当前的 patch_size ---
        # 使用 .get() 方法，如果找不到 key 则使用默认值
        current_config = dataset_configs.get(ds_name, {"patch_size": default_patch_size})
        current_patch_size = current_config["patch_size"]

        ds_path = os.path.join(base_dir, ds_name)
        image_dir = os.path.join(ds_path, "image")
        
        # 1. 检查数据集是否存在且不为空
        if not os.path.exists(image_dir) or len(os.listdir(image_dir)) == 0:
            print(f"Skipping {ds_name}: Directory empty or not found.")
            continue
        
        print(f"\n{'='*20} Processing Dataset: {ds_name} {'='*20}")
        print(f" Current patch size = {current_patch_size} for dataset {ds_name}")
        
        # 获取该数据集下所有样本 ID (通过 image 文件夹下的 png 文件名获取)
        sample_ids = [f.replace(".png", "") for f in os.listdir(image_dir) if f.endswith(".png")]
        
        # 为每个数据集创建独立的输出目录
        ds_output_dir = os.path.join(output_root, ds_name)
        os.makedirs(ds_output_dir, exist_ok=True)

        for sample_id in sample_ids:
            print(f"--- Processing Sample: {sample_id} ---")
            
            # 构建文件路径
            wsi_path = os.path.join(image_dir, f"{sample_id}.png")
            coord_path = os.path.join(ds_path, "coord", f"{sample_id}_coord.csv")
            anno_path = os.path.join(ds_path, "annotation", f"{sample_id}_anno.csv")
            
            # 检查文件完整性
            if not (os.path.exists(coord_path) and os.path.exists(anno_path)):
                print(f"Warning: Metadata missing for {sample_id}, skipping.")
                continue

            # 2. 读取并合并坐标与标注
            # 假设 coord 的第一列和 anno 的第一列（name）是对齐的
            df_coord = pd.read_csv(coord_path)
            df_anno = pd.read_csv(anno_path)
            
            # 统一列名：确保第一列叫 'name'，如果是空列名则重命名
            if df_coord.columns[0].startswith('Unnamed') or df_coord.columns[0] == "":
                df_coord.rename(columns={df_coord.columns[0]: 'name'}, inplace=True)
            
            # 合并数据以确保 row 对齐 (基于 name 列)
            df_merged = pd.merge(df_coord, df_anno[['name', 'label']], on='name')

            # 3. 数据加载
            dataset = SpatialSpotDataset(df_merged, wsi_path, patch_size=current_patch_size, transform=transform)
            dataloader = DataLoader(dataset, 
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=4,
                                    pin_memory=True)

            all_features, all_coords, all_spot_ids, all_true_labels = [], [], [], []

            with torch.no_grad():
                for batch_imgs, batch_coords, batch_spot_ids, batch_true_labels in tqdm(dataloader, desc=f"Extracting {sample_id}"):
                    if torch.cuda.is_available():
                        batch_imgs = batch_imgs.cuda()
                    
                    features = model(batch_imgs)
                    all_features.append(features.cpu().numpy())
                    
                    coords_np = np.stack([c.numpy() for c in batch_coords], axis=1)
                    all_coords.append(coords_np)
                    all_spot_ids.extend(batch_spot_ids)
                    all_true_labels.extend(batch_true_labels)

            # 4. 保存结果
            if len(all_features) > 0:
                final_features = np.concatenate(all_features, axis=0)
                final_coords = np.concatenate(all_coords, axis=0)
                final_spot_ids = np.array(all_spot_ids, dtype=h5py.string_dtype(encoding='utf-8'))
                final_true_labels = np.array(all_true_labels, dtype=h5py.string_dtype(encoding='utf-8'))

                save_path = os.path.join(ds_output_dir, f"{sample_id}_gigapath_features.h5")
                with h5py.File(save_path, 'w') as f:
                    f.create_dataset('features', data=final_features)
                    f.create_dataset('coords', data=final_coords)
                    f.create_dataset('spot_ids', data=final_spot_ids)
                    f.create_dataset('true_labels', data=final_true_labels)
                print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()