import numpy as np
from scipy.ndimage import label
import scipy 
import open3d as o3d
from classes.PCD import PCD
import matplotlib.pyplot as plt

def load_pcd(file_path):
    pc_data = PCD()
    pc_data.open(file_path, verbose=False)  # verbose - логирование
    points = np.asarray(pc_data.points[:, :3])  # Assuming points[:, :3] contains the x, y, z coordinates
    return points, pc_data

def voxelize_points(points, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size) # создаем воксельную сетку из облака точек
    return voxel_grid

def voxel_grid_to_numpy(voxel_grid):
    voxels = voxel_grid.get_voxels()
    voxel_indices = np.array([v.grid_index for v in voxels])
    return voxel_indices

def label_connected_components(voxel_indices, connectivity=26):
    shape = voxel_indices.max(axis=0) + 1
    voxel_array = np.zeros(shape, dtype=bool)
    voxel_array[tuple(voxel_indices.T)] = True
    labeled_array, num_features = scipy.ndimage.label(voxel_array, structure=np.ones((3, 3, 3)) if connectivity == 26 else np.eye(3))
    return labeled_array, num_features

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
    

if __name__ == "__main__":
    voxel_size = 0.1
    connectivity = 26
    input_path = "trees.pcd"  # "/path/to/input.las", or "/path/to/input.pcd"
    output_path = "trees_voxeled.pcd"  # "/path/to/output.pcd" !only *.pcd!

    # Шаг 1: Загрузка облака точек
    points, pc_data = load_pcd(input_path)
    
    # Шаг 2: Вокселизация
    voxel_grid = voxelize_points(points, voxel_size)
    
    # Шаг 3: Преобразование вокселей в numpy массив
    voxel_indices = voxel_grid_to_numpy(voxel_grid)
    
    # Шаг 4: Нахождение связанных компонент
    labeled_array, num_features = label_connected_components(voxel_indices, connectivity)
    
    # Шаг 5: Маппинг меток вокселей к исходным точкам
    labels = map_voxel_labels_to_points(points, voxel_indices, labeled_array, voxel_size)


    # Шаг 6: Создание облака точек с метками (для визуализации, если нужно)
    labeled_pcd, _ = create_labeled_point_cloud(voxel_indices, labeled_array)
    
    # Шаг 7: Сохранение результата в PCD файл
    save_labeled_pcd(pc_data, labels, output_path)
    print(f"Saved labeled point cloud to {output_path} with {num_features} connected components.")
