import os
import struct
import math
import numpy as np
import scipy.interpolate

# Import voxelize_occupy directly from data_utils
from data_utils import voxelize_occupy

sorted_vertical_angles = [
    -30.67, -29.33, -28.0, -26.67, -25.33, -24.0, -22.67, -21.33, -20.0,
    -18.67, -17.33, -16.0, -14.67, -13.33, -12.0, -10.67, -9.3299999, -8.0,
    -6.6700001, -5.3299999, -4.0, -2.6700001, -1.33, 0.0, 1.33, 2.6700001, 4.0,
    5.3299999, 6.6700001, 8.0, 9.3299999, 10.67
]

def carmen_normalize_theta(theta):
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta < -math.pi:
        theta += 2.0 * math.pi
    return theta

def parse_carmen_pointcloud(file_path):
    points = []
    shot_struct = struct.Struct('<d 32H 32B')
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
        
    num_shots = len(data) // shot_struct.size
    
    for i in range(num_shots):
        offset = i * shot_struct.size
        unpacked = shot_struct.unpack_from(data, offset)
        
        angle = unpacked[0]
        distances = unpacked[1:33]
        intensities = unpacked[33:65]
        
        for j in range(32):
            v_deg = sorted_vertical_angles[j]
            v = carmen_normalize_theta(math.radians(v_deg))
            h = carmen_normalize_theta(math.radians(-angle))
            
            range_m = distances[j] / 500.0
            intensity = intensities[j]
            
            if 0 < range_m < 200:
                x = range_m * math.cos(v) * math.cos(h)
                y = range_m * math.cos(v) * math.sin(h)
                z = range_m * math.sin(v)
                
                # NuScenes coordinates: X right, Y forward, Z up
                nusc_x = -y
                nusc_y = x
                nusc_z = z
                
                points.append([nusc_x, nusc_y, nusc_z, intensity, j])
                
    if len(points) == 0:
        return None
    # MotionNet expects shape (5, N) -> x, y, z, intensity, ring
    return np.array(points, dtype=np.float32).T

def load_carmen_log(log_path):
    lidar_frames = []
    velocities = []
    
    log_dir = os.path.dirname(log_path)
    
    print("Reading CARMEN log...")
    with open(log_path, 'r') as f:
        for line in f:
            if line.startswith('ROBOTVELOCITY_ACK'):
                # ROBOTVELOCITY_ACK tv rv timestamp host logger_timestamp
                parts = line.strip().split()
                if len(parts) >= 4:
                    tv = float(parts[1])
                    rv = float(parts[2])
                    ts = float(parts[3])
                    velocities.append((ts, tv, rv))
            elif line.startswith('VELODYNE_PARTIAL_SCAN_IN_FILE'):
                # VELODYNE_PARTIAL_SCAN_IN_FILE path num_shots timestamp host logger_timestamp
                parts = line.strip().split()
                if len(parts) >= 4:
                    rel_path = parts[1]
                    ts = float(parts[3])
                    # Fix path
                    full_path = os.path.join(log_dir, os.path.basename(log_path) + rel_path)
                    lidar_frames.append((ts, full_path))
                    
    # Sort by timestamp
    velocities.sort(key=lambda x: x[0])
    lidar_frames.sort(key=lambda x: x[0])
    
    print(f"Found {len(lidar_frames)} lidar frames and {len(velocities)} velocity messages.")
    return lidar_frames, velocities

def compute_odometry(velocities):
    poses = []
    x, y, theta = 0.0, 0.0, 0.0
    
    if len(velocities) == 0:
        return poses
        
    last_t = velocities[0][0]
    poses.append((last_t, x, y, theta))
    
    for i in range(1, len(velocities)):
        t, tv, rv = velocities[i]
        dt = t - last_t
        x += tv * math.cos(theta) * dt
        y += tv * math.sin(theta) * dt
        theta += rv * dt
        theta = carmen_normalize_theta(theta)
        
        poses.append((t, x, y, theta))
        last_t = t
        
    return poses

def get_pose_at_time(t, poses_arr):
    idx = np.searchsorted(poses_arr[:, 0], t)
    if idx == 0:
        return poses_arr[0, 1:]
    elif idx == len(poses_arr):
        return poses_arr[-1, 1:]
    else:
        t0, x0, y0, th0 = poses_arr[idx-1]
        t1, x1, y1, th1 = poses_arr[idx]
        alpha = (t - t0) / (t1 - t0) if t1 != t0 else 0
        
        x = x0 + alpha * (x1 - x0)
        y = y0 + alpha * (y1 - y0)
        diff = carmen_normalize_theta(th1 - th0)
        th = carmen_normalize_theta(th0 + alpha * diff)
        return np.array([x, y, th])

def transform_pc(pc, pose_curr, pose_past):
    x_c, y_c, th_c = pose_curr
    x_p, y_p, th_p = pose_past
    
    c_x = pc[1, :]
    c_y = -pc[0, :]
    
    cos_p = math.cos(th_p)
    sin_p = math.sin(th_p)
    g_x = cos_p * c_x - sin_p * c_y + x_p
    g_y = sin_p * c_x + cos_p * c_y + y_p
    
    dx = g_x - x_c
    dy = g_y - y_c
    cos_c = math.cos(-th_c)
    sin_c = math.sin(-th_c)
    curr_c_x = cos_c * dx - sin_c * dy
    curr_c_y = sin_c * dx + cos_c * dy
    
    n_x = -curr_c_y
    n_y = curr_c_x
    
    out_pc = pc.copy()
    out_pc[0, :] = n_x
    out_pc[1, :] = n_y
    return out_pc

def get_bev(data_dict):
    voxel_size = (0.25, 0.25, 0.4)
    area_extents = np.array([[-32.0, 32.0], [-32.0, 32.0], [-3.0, 2.0]])
    padded_voxel_points_list = []
    num_sweeps = data_dict["num_sweeps"]
    for i in range(num_sweeps):
        pc = data_dict[f"pc_{i}"].T[:, :4]  # N x 4
        res = voxelize_occupy(pc, voxel_size=voxel_size, extents=area_extents, return_indices=False)
        padded_voxel_points_list.append(res)
    return np.stack(padded_voxel_points_list, axis=0).astype(bool)

def process_carmen_log(log_path, output_dir, max_frames=50, start_frame=0):
    os.makedirs(output_dir, exist_ok=True)
    
    lidar_frames, velocities = load_carmen_log(log_path)
    if not lidar_frames:
        print("No lidar frames found.")
        return
        
    poses = compute_odometry(velocities)
    poses_arr = np.array(poses)
    
    nsweeps_back = 4
    num_sweeps = nsweeps_back + 1
    
    print(f"Generating BEV maps to {output_dir}")
    
    processed = 0
    # Process each frame that has enough past sweeps, starting from start_frame
    start_idx = max(nsweeps_back, start_frame)
    for i in range(start_idx, len(lidar_frames)):
        if processed >= max_frames:
            break
            
        curr_t, curr_path = lidar_frames[i]
        pose_curr = get_pose_at_time(curr_t, poses_arr)
        
        data_dict = {}
        data_dict["num_sweeps"] = num_sweeps
        
        valid = True
        
        # Order is [past_4, past_3, past_2, past_1, current]
        # In MotionNet it goes [oldest -> newest] for prediction?
        # Actually `gen_data.py` lines 336: `pc_list = tmp_pc_list_1 + tmp_pc_list_2`
        # `tmp_pc_list_1` is `pc_list[0:num_past_sweeps][::-1]`. 
        # Wait, if `data_dict['pc_0']` was the current frame, then `pc_list[0]` is current, `pc_list[1]` is past_1...
        # Then `[::-1]` makes it `[past_4, past_3, past_2, past_1, current]`.
        # So our `get_bev` assumes the input dict is ALREADY sorted [oldest -> newest].
        for j in range(num_sweeps):
            # frame_idx goes from (i - 4) to i
            frame_idx = i - nsweeps_back + j
            past_t, past_path = lidar_frames[frame_idx]
            pose_past = get_pose_at_time(past_t, poses_arr)
            
            pc = parse_carmen_pointcloud(past_path)
            if pc is None:
                valid = False
                break
                
            if j < num_sweeps - 1:
                pc = transform_pc(pc, pose_curr, pose_past)
                
            data_dict[f"pc_{j}"] = pc
            
        if not valid:
            continue
            
        spatial_features = get_bev(data_dict)
        
        out_name = f"{curr_t:.6f}.npy"
        np.save(os.path.join(output_dir, out_name), spatial_features)
        
        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed}/{max_frames} frames")
            
    print("Done!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, required=True, help='Path to CARMEN log txt file')
    parser.add_argument('--out_dir', type=str, default='carmen-preprocessed', help='Output directory for NPY files')
    parser.add_argument('--max_frames', type=int, default=50, help='Max frames to process')
    parser.add_argument('--start_frame', type=int, default=0, help='Starting frame index to skip uninteresting parts')
    args = parser.parse_args()
    
    process_carmen_log(args.log, args.out_dir, max_frames=args.max_frames, start_frame=args.start_frame)
