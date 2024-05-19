import numpy as np
import pprint
from time import time
from .PCD_UTILS import PCD_UTILS
import laspy

class PCD:
    def __init__(self, points = None, intensity = None):
        self.points = points
        self.intensity = intensity
 
    def save(self, file_path, verbose = False):
        """ save .pcd with intensity """
        if verbose:
            print(f"Saving .pcd file ...")
        start = time()
        self.points = np.asarray(self.points)
        if self.intensity == None:
            self.intensity = np.full(self.points.shape[0], 0)
        
        dt = np.c_[self.points, self.intensity] # по сути добавляет колонку intensity нашему массиву точек
        dt = np.array(dt, dtype=np.float32)
        new_cloud = PCD_UTILS.make_xyz_intensity_point_cloud(dt)
        if verbose:
            pprint.pprint(new_cloud.get_metadata())
            print(dt, dt.shape)
        new_cloud.save_pcd(file_path, 'binary')
        if verbose:
            end = time()-start
            print(f"Time saving: {end:.3f} s")

    def open(self, file_path, mode = 'intensity', verbose = False):
        """ open .pcd with intensity """
        if file_path.endswith('.pcd'):
            if verbose:
                start = time()
                print(f"Opening .pcd file ...")
            data, ix, ii, ir = PCD_UTILS.PCD_OPEN_X_INT_RGB(file_path, verbose)
            if verbose:
                end = time()-start
                print(f"Time reading: {end:.3f} s")
                start = time()
            points = data[:,ix:ix + 3]
            if mode == 'rgb':
                intensity = np.asarray(data[:,ir]) if ir is not None else None
            else:
                intensity = np.asarray(data[:,ii]) if ii is not None else None
            intensity = np.nan_to_num(intensity)
            if verbose:
                end = time()-start
                print(f"Time stacking data: {end:.3f} s")
            self.points, self.intensity = points, intensity

        if file_path.endswith('.las'):
            if verbose:
                start = time()
                print(f"Opening .las file ...")
            las = laspy.read(file_path)
            if verbose:
                end = time()-start
                print(f"Time reading: {end:.3f} s")
                start = time()
            points = np.vstack([las.points.x, las.points.y, las.points.z]).transpose()
            intensity = np.asarray(las.intensity, dtype = np.int32)
            intensity = np.nan_to_num(intensity)
            if verbose:
                end = time()-start
                print(f"Time stacking data: {end:.3f} s")
            self.points, self.intensity = points, intensity

