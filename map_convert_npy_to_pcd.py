import open3d as o3d
import numpy as np
import sys

if __name__=="__main__":
    npy_name = sys.argv[1]
    pcd_name = sys.argv[2]

    if npy_name is None or pcd_name is None:
        print('usage: npy_file pcd_file')

    points = np.load(npy_name)

    pcd = o3d.t.geometry.PointCloud()
    
    pcd.point['positions'] = o3d.core.Tensor(points)
    
    o3d.t.io.write_point_cloud(pcd_name, pcd, write_ascii=True)