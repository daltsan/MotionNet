import os
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio

from model import MotionNet
from data.data_utils import voxelize_occupy

color_map = {0: "c", 1: "m", 2: "k", 3: "y", 4: "r"}
cat_names = {0: "bg", 1: "bus", 2: "ped", 3: "bike", 4: "other"}

def vis_carmen_data(data_dir, model_path, img_save_dir):
    os.makedirs(img_save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading model...")
    model = MotionNet(out_seq_len=20, motion_category_num=2, height_feat_size=13)
    model = nn.DataParallel(model)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    print(f"Found {len(npy_files)} files to process.")
    
    border_meter = 4
    border_pixel = border_meter * 4
    voxel_size = (0.25, 0.25, 0.4)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    
    for idx, fpath in enumerate(npy_files):
        print(f"Processing {idx+1}/{len(npy_files)}: {os.path.basename(fpath)}")
        
        # Load BEV feature (5, 256, 256, 13)
        padded_voxel_points = np.load(fpath)
        inp = torch.from_numpy(padded_voxel_points).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            disp_pred, cat_pred, motion_pred = model(inp)
            
            # disp_pred is [20, 2, 256, 256]
            disp_pred = disp_pred.cpu().numpy()
            disp_pred = np.transpose(disp_pred, (0, 2, 3, 1)) # [20, 256, 256, 2]
            
            cat_pred = np.squeeze(cat_pred.cpu().numpy(), 0) # [5, H, W]
            
            motion_pred_numpy = motion_pred.cpu().numpy()
            motion_pred_numpy = np.argmax(motion_pred_numpy, axis=1)[0] # [H, W]
            
        # Cumulative displacement for adjacent frame prediction
        for c in range(1, disp_pred.shape[0]):
            disp_pred[c, ...] = disp_pred[c, ...] + disp_pred[c - 1, ...]
            
        # Mask using motion state
        motion_mask = motion_pred_numpy == 0
        
        # Determine non-empty map from the current frame (index 4 out of 5 sweeps)
        curr_voxel = padded_voxel_points[-1] # [256, 256, 13]
        non_empty_map = np.any(curr_voxel, axis=-1).astype(np.float32)
        
        cat_pred_numpy = np.argmax(cat_pred, axis=0)
        cat_mask = np.logical_and(cat_pred_numpy == 0, non_empty_map == 1)
        
        cat_weight_map = np.ones_like(motion_pred_numpy, dtype=np.float32)
        cat_weight_map[motion_mask] = 0.0
        cat_weight_map[cat_mask] = 0.0
        cat_weight_map = cat_weight_map[:, :, np.newaxis]
        
        disp_pred = disp_pred * cat_weight_map
        
        cat_pred_class = np.argmax(cat_pred, axis=0) + 1
        cat_pred_class = (cat_pred_class * non_empty_map).astype(int)
        
        # Plot
        ax[0].clear()
        ax[1].clear()
        
        # 1. Lidar Plot
        # Get point coordinates from voxel
        y_idx, x_idx = np.where(non_empty_map == 1)
        x_pts = x_idx * voxel_size[0] - 32.0
        y_pts = y_idx * voxel_size[1] - 32.0
        
        ax[0].scatter(x_pts, y_pts, c='blue', s=1)
        ax[0].set_xlim(-28, 28)
        ax[0].set_ylim(-28, 28)
        ax[0].axis("off")
        ax[0].set_aspect("equal")
        ax[0].title.set_text("LIDAR current sweep")
        
        # 2. Prediction Quiver
        field_pred = disp_pred[-1] # 20th frame
        field_pred_norm = np.linalg.norm(field_pred, ord=2, axis=-1)
        thd_mask = field_pred_norm <= 0.4
        field_pred[thd_mask, :] = 0
        
        idx_x = np.arange(field_pred.shape[0])
        idx_y = np.arange(field_pred.shape[1])
        idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing="ij")
        
        for k in range(len(color_map)):
            mask_pred = cat_pred_class == (k + 1)
            
            X_pred = idx_x[mask_pred]
            Y_pred = idx_y[mask_pred]
            U_pred = field_pred[:, :, 0][mask_pred] / voxel_size[0]
            V_pred = field_pred[:, :, 1][mask_pred] / voxel_size[1]
            
            if len(X_pred) > 0:
                ax[1].quiver(X_pred, Y_pred, U_pred, V_pred, angles="xy", scale_units="xy", scale=1, color=color_map[k])
                
        ax[1].set_xlim(border_pixel, field_pred.shape[0] - border_pixel)
        ax[1].set_ylim(border_pixel, field_pred.shape[1] - border_pixel)
        ax[1].set_aspect("equal")
        ax[1].title.set_text("Prediction")
        ax[1].axis("off")
        
        plt.savefig(os.path.join(img_save_dir, f"{idx}.png"))
        
    print("Generating video...")
    save_gif_path = os.path.join(img_save_dir, "result.gif")
    with imageio.get_writer(save_gif_path, mode="I", fps=10) as writer:
        for i in range(len(npy_files)):
            image_file = os.path.join(img_save_dir, f"{i}.png")
            image = imageio.imread(image_file)
            writer.append_data(image)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/davi/Ufes/robprob/carmen-preprocessed')
    parser.add_argument('--model_path', type=str, default='/home/davi/Ufes/robprob/pre-tained-model/model.pth')
    parser.add_argument('--img_save_dir', type=str, default='/home/davi/Ufes/robprob/carmen_results')
    args = parser.parse_args()
    
    vis_carmen_data(args.data_dir, args.model_path, args.img_save_dir)
