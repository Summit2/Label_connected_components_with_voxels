import numpy as np
from scipy.ndimage import label
import scipy 
import open3d as o3d
from classes.PCD import PCD
import matplotlib.pyplot as plt
import cc3d #connected-components
from PIL import Image
import PIL

def load_pcd(file_path):
    pc_data = PCD()
    pc_data.open(file_path, verbose=False)  # verbose - логирование
    points = np.asarray(pc_data.points[:, :3])  # Assuming points[:, :3] contains the x, y, z coordinates
    return points, pc_data

def voxelize_points(points, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size) # создаем воксельную сетку из облака точек
    # o3d.visualization.draw_geometries([voxel_grid]) #можно промежуточно посмотреть на воксели
    return voxel_grid

def voxel_grid_to_numpy(voxel_grid):
    voxels = voxel_grid.get_voxels()
    voxel_indices = np.array([v.grid_index for v in voxels])
    return voxel_indices


def label_connected_components_cc3d(voxel_indices, connectivity, delta):
# надо понять, что подавать, чтобы получать нормальный результат
    labels_out , N  = cc3d.connected_components(voxel_indices, connectivity=connectivity, return_N = True, delta=delta)
    

    return labels_out , N


def encode_voxels_to_cc3d_format(voxel_indices):
    '''
    voxel_indices.shape = (x, 3)  
    '''
# пример
#     filled = np.array([       #y0        #y1      ...
#                             [[0, 0, 1],[1, 0, 1],[1, 0, 1],[0, 0, 1],[0, 0, 1],[0, 0, 1]], #x0
#                             [[1, 0, 0],[1, 0, 0],[1, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0]], #x1
#                             [[1, 0, 0],[1, 0, 0],[1, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0]],
#                             [[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 1, 0],[0, 1, 0],[0, 1, 0]],
#                             [[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 1, 0],[0, 1, 0],[0, 1, 0]],
#                             [[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 1, 0],[0, 1, 0],[0, 1, 0]]   
#                             # а внутри координаты, сколько брать по z
# ])

    
    x_max = np.max(voxel_indices[:, 0])
    y_max = np.max(voxel_indices[:, 1])
    z_max = np.max(voxel_indices[:, 2])
    
    labels = np.zeros((x_max+1,y_max+1,z_max+1)) 

    for x,y,z in voxel_indices:
        labels[x,y,z] = 1
    
    return labels
   


def decode_cc3d_to_voxels_with_class_labels(cc3d_input,voxel_indices):

    
    class_labels = np.zeros(voxel_indices.shape[0], dtype=np.int32)

    for i, xyz in enumerate(voxel_indices):
        
        class_labels[i] = cc3d_input[xyz[0],xyz[1],xyz[2]]
    return class_labels



def map_voxel_labels_to_points(points, voxel_indices, labeled_array, voxel_size):
    voxel_indices_dict = {tuple(idx): label for idx, label in zip(voxel_indices, labeled_array[tuple(voxel_indices.T)])}
    labels = []
    for point in points:
        voxel_idx = tuple((point / voxel_size).astype(int))
        labels.append(voxel_indices_dict.get(voxel_idx, 0))  # Default label to 0 if not found
    return np.array(labels)

def create_labeled_point_cloud(voxel_indices, labeled_array):
    labels = labeled_array[tuple(voxel_indices.T)]
    colors = plt.get_cmap('tab20')(labels / labels.max())
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_indices)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd, labels


def save_labeled_pcd(original_pc_data, labels, output_path):
    # Ensure labels are of the correct dtype (uint32)
    labels = labels.astype(np.float64)


    # Create a structured array to hold the original points and their corresponding labels
    labeled_data = np.zeros(len(original_pc_data.points), dtype=[('x', 'float32'), ('y', 'float32'), ('z', 'float32'), ('label', 'float32')])
    
    labeled_data['x'] = original_pc_data.points[:, 0]
    labeled_data['y'] = original_pc_data.points[:, 1]
    labeled_data['z'] = original_pc_data.points[:, 2]
    labeled_data['label'] = labels  # Ensure labels fit the original data length

    labeled_data = np.array([ [ original_pc_data.points[i, 0], original_pc_data.points[i, 1], original_pc_data.points[i,2], labels[i]] 
                             for i in range(len(original_pc_data.points))], dtype='float32')


    # labeled_data = np.vstack([np.array(original_pc_data.points[:, 0]),np.array(original_pc_data.points[:, 1]),np.array(original_pc_data.points[:, 2]),labels])

    print(labeled_data.dtype)
    new_pc_data = PCD()
    new_pc_data.points = labeled_data  # Assign the structured array to new_pc_data.points
    print(labeled_data)
    # Save the labeled point cloud
    new_pc_data.save(output_path, verbose=True)

def label_to_rgb(label, num_classes):
    """
    Convert a class label to an RGB color.
    """
    rgb = np.zeros((num_classes, 3))
    for i in range(num_classes):
        rgb[i] = np.random.rand(3)  # Generate random RGB values for demonstration
    return rgb[label]


if __name__ == "__main__":
    voxel_size = 0.10023
    connectivity = 26
    input_path = "trees.pcd"  # "/path/to/input.las", or "/path/to/input.pcd"
    output_path = "trees_voxeled.pcd"  # "/path/to/output.pcd" !only *.pcd!

 
    points, pc_data = load_pcd(input_path)
    

    voxel_grid = voxelize_points(points, voxel_size)
    

    voxel_indices = voxel_grid_to_numpy(voxel_grid)
    
    
    
    
    cc3d_labels = encode_voxels_to_cc3d_format(voxel_indices)
    
    
    labels_out,N_components = label_connected_components_cc3d(cc3d_labels, connectivity = connectivity, delta = 0)

   
 
    print('Number of components detected:',N_components)

    class_labels = decode_cc3d_to_voxels_with_class_labels(labels_out,voxel_indices)   
 
    # save_labeled_pcd(pc_data, labels, output_path)
    # print(f"Saved labeled point cloud to {output_path} with {num_features} connected components.")
    
   
    cmap = plt.get_cmap('viridis')
    rgb_colors = cmap(class_labels) # RGBA
    
    
    pcd = o3d.geometry.PointCloud()
    
    

    pcd.points = o3d.utility.Vector3dVector(voxel_indices)
    pcd.colors = o3d.utility.Vector3dVector(rgb_colors[:, :3])
    
    # print(np.column_stack((rgb_colors[:, 0], rgb_colors[:, 1], rgb_colors[:, 2])))
    o3d.visualization.draw_geometries([pcd])
    
    