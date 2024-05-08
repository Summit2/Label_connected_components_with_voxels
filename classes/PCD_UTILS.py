import numpy as np
import pprint
from time import time
from pypcd import pypcd
import math

class PCD_UTILS:
    def make_xyz_intensity_point_cloud(xyz_intensity, metadata=None):
        """ Make a pointcloud object from xyz array.
        xyz array is assumed to be float32.
        intensity is assumed to be encoded as float32 according to pcl conventions.
        """
        md = {'version': .7,
            'fields': ['x', 'y', 'z', 'Intensity'],
            'count': [1, 1, 1, 1],
            'width': len(xyz_intensity),
            'height': 1,
            'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            'points': len(xyz_intensity),
            'type': ['F', 'F', 'F', 'F'],
            'size': [4, 4, 4, 4],
            'data': 'binary'}
        if xyz_intensity.dtype != np.float32:
            raise ValueError('array must be float32')
        if metadata is not None:
            md.update(metadata)
        pc_data = xyz_intensity.view(np.dtype([('x', np.float32),
                                        ('y', np.float32),
                                        ('z', np.float32),
                                        ('Intensity', np.float32)])).squeeze()
        pc = pypcd.PointCloud(md, pc_data)
        return pc


    def PCD_OPEN_X_INT_RGB(file_path, verbose = False):
        """ Return data, indexes of fields 'x', 'Intensity' and 'rgb'
        Input:
            file_path: string, /path/to/file/example.foo
            verbose: boolean, enable print info
        Return:
            new_cloud_data: data from file
            ix, ii, ir = integer, indexes of fields 'x', 'Intensity' and 'rgb'
        """
        start = time()
        cloud = pypcd.PointCloud.from_path(file_path)
        new_cloud_data = cloud.pc_data.copy()
        if verbose:
            print(f"Opening {file_path}")
            pprint.pprint(cloud.get_metadata())
        new_cloud_data = cloud.pc_data.view(np.float32).reshape(cloud.pc_data.shape + (-1,))
        if verbose:
            print(new_cloud_data)
            print(f"Shape: {new_cloud_data.shape}")
            end = time()-start
            print(f"Time opening: {end:.3f} s")
        try:
            ii = cloud.get_metadata()["fields"].index('Intensity')
        except ValueError:
            ii = None
        try:
            ir = cloud.get_metadata()["fields"].index('rgb')
        except ValueError:
            ir = None
        ix = cloud.get_metadata()["fields"].index('x')
        return new_cloud_data, ix, ii, ir 
 
    def shift(points, x, y, z):
        """ Return shifted data
        Input:
            points: origin data
            x, y, z: shift values along the X, Y and Z axes, respectively.
        Return:
            points + shift_matrix: shifted data
        """
        x_shift = np.full((points.shape[0],1), x)
        y_shift = np.full((points.shape[0],1), y)
        z_shift = np.full((points.shape[0],1), z)
        if points.shape[1] == 3:
            shift_matrix = np.concatenate([x_shift, y_shift, z_shift], axis=1)
        elif points.shape[1] == 2:
            shift_matrix = np.concatenate([x_shift, y_shift], axis=1)
        elif points.shape[1] == 1:
            shift_matrix = np.concatenate([x_shift], axis=1)
        return points + shift_matrix
    
    def rotate_points(points, rotation_angle):
        """ Return rotated data
        Input:
            points: origin data
            rotation_angle: rotation angle in radians
        Return:
            rotated_points: rotated data
        """
        rotated_points = []
        for point in points:
            xp = math.cos(rotation_angle) * point[0] - math.sin(rotation_angle) * point[1]
            yp = math.sin(rotation_angle) * point[0] + math.cos(rotation_angle) * point[1]
            rotated_points.append([xp, yp, point[2]])
        return rotated_points



