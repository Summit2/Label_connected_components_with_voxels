import numpy as np
from scipy.ndimage import label
import scipy 
import open3d as o3d
from classes.PCD import PCD
from classes.PCD_UTILS import PCD_UTILS
import matplotlib.pyplot as plt
import cc3d #connected-components
from PIL import Image
import PIL

def load_pcd(file_path):
    pc_data = PCD()
    pc_data.open(file_path, verbose=False)  # verbose - logging
    points = np.asarray(pc_data.points[:, :3])  # Assuming points[:, :3] contains the x, y, z coordinates
    return points, pc_data

def voxelize_points(points, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size) # create voxel grid from point cloud
    return voxel_grid

def voxel_grid_to_numpy(voxel_grid):
    voxels = voxel_grid.get_voxels()
    voxel_indices = np.array([v.grid_index for v in voxels])
    return voxel_indices

def encode_voxels_to_cc3d_format(voxel_indices):
    x_max = np.max(voxel_indices[:, 0])
    y_max = np.max(voxel_indices[:, 1])
    z_max = np.max(voxel_indices[:, 2])
    
    labels = np.zeros((x_max+1, y_max+1, z_max+1)) 

    for x, y, z in voxel_indices:
        labels[x, y, z] = 1
    
    return labels

def decode_cc3d_to_voxels_with_class_labels(cc3d_input, voxel_indices):
    class_labels = np.zeros(voxel_indices.shape[0], dtype=np.int32)

    for i, xyz in enumerate(voxel_indices):
        class_labels[i] = cc3d_input[xyz[0], xyz[1], xyz[2]]
    return class_labels

def label_connected_components_with_voxels(pcd_points, voxel_size, connectivity):
    
    voxel_grid = voxelize_points(pcd_points, voxel_size)
    voxel_indices = voxel_grid_to_numpy(voxel_grid)
       
    cc3d_labels = encode_voxels_to_cc3d_format(voxel_indices)
    labels_out, N_components = cc3d.connected_components(cc3d_labels, connectivity=connectivity, return_N=True)
 
    print('Number of components detected:', N_components)

    class_labels = decode_cc3d_to_voxels_with_class_labels(labels_out, voxel_indices)   
   
    cmap = plt.get_cmap('Paired')
    class_labels_RGB = cmap(class_labels)[:,:3]  # RGBA -> RGB


    return voxel_indices, class_labels_RGB, voxel_grid, labels_out


def visualize_voxel_indices(voxel_indices, colors = np.array([])):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_indices)
    
    if colors.shape[0]!=0:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.colors = o3d.cpu.pybind.utility.Vector3dVector(np.zeros(shape = (voxel_indices.shape[0],3)))

    o3d.visualization.draw_geometries([pcd])


def assign_labels_from_voxels_to_original_points(original_points, voxel_grid, labels_out):
    voxel_size = voxel_grid.voxel_size
    origin = voxel_grid.origin
    
    labeled_points = []
    for point in original_points:
        voxel_index = ((point - origin) / voxel_size).astype(int)
        label = labels_out[tuple(voxel_index)]
        labeled_points.append((point, label))
    
    return labeled_points

def visualize_labeled_points(labeled_points):
    points = np.array([p[0] for p in labeled_points])
    labels = np.array([p[1] for p in labeled_points])
    
    cmap = plt.get_cmap('Paired')
    colors = cmap(labels)[:,:3]  # RGBA -> RGB
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([pcd])

    return pcd




    
if __name__ == "__main__":
    
    input_path = "trees.pcd"  # "/path/to/input.las", or "/path/to/input.pcd"
    output_path = "trees_voxeled.pcd"  # "/path/to/output.pcd" !only *.pcd!
    points, pc_data = load_pcd(input_path)
    

    voxel_size = 0.1005
    connectivity = 26
    voxel_indices, class_labels_RGB, voxel_grid, labels_out = label_connected_components_with_voxels(points, voxel_size, connectivity)
    
  
    labeled_points = assign_labels_from_voxels_to_original_points(points, voxel_grid, labels_out) #конвертируем воксели обратно в точки, уже с цветом
    
    visualize_labeled_points(labeled_points)



    