from classes.PCD import PCD
from classes.PCD_UTILS import PCD_UTILS
import math

def shift_rotate(input_path, output_path, shift_x, shift_y, shift_z, rotation_angle):
    pc_data = PCD()
    pc_data.open(input_path, verbose = True) # verbose = False
    pc_data.points = PCD_UTILS.shift(pc_data.points, shift_x, shift_y, shift_z)
    pc_data.points = PCD_UTILS.rotate_points(pc_data.points, rotation_angle)
    pc_data.save(output_path, verbose = True) # verbose = False

def label_connected_components(input_path, output_path):
    pc_data = PCD()
    pc_data.open(input_path, verbose = True) # verbose = False
    # 
    
    pc_data.save(output_path, verbose = True) # verbose = False

if __name__ == "__main__" :
    input_path = "input.pcd"               #"/path/to/input.las", or "/path/to/input.pcd"
    output_path = "output.pcd"             #"/path/to/output.pcd" !only *.pcd!
    shift_x, shift_y, shift_z = 10, 10, 0
    rotation_angle = math.pi/4
    shift_rotate(input_path, output_path, shift_x, shift_y, shift_z, rotation_angle)