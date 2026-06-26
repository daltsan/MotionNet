import struct
import numpy as np
import math
import os

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
    # format per 32-laser shot: double angle, 32 short distances, 32 byte intensities
    shot_struct = struct.Struct('<d 32H 32B')
    
    with open(file_path, 'rb') as f:
        data = f.read()
        
    num_shots = len(data) // shot_struct.size
    
    for i in range(num_shots):
        offset = i * shot_struct.size
        unpacked = shot_struct.unpack_from(data, offset)
        
        angle = unpacked[0]
        distances = unpacked[1:33]
        intensities = unpacked[33:65]
        
        for j in range(32):
            v_deg = sorted_vertical_angles[j]
            # Some scripts call arrange_velodyne_vertical_angles_to_true_position
            # which modifies the vertical angles. For now we use the sorted ones.
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
                
    return np.array(points, dtype=np.float32)

if __name__ == '__main__':
    import sys
    pcd_file = sys.argv[1] if len(sys.argv) > 1 else 'example.pointcloud'
    pc = parse_carmen_pointcloud(pcd_file)
    print(f"Loaded {len(pc)} points")
    print("Sample points (x, y, z, intensity, ring):")
    print(pc[:5])
