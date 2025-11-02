import os
import subprocess

def initialize(name, folder):
    input_path = os.path.join(folder, name)
    command = f"python ./initialize.py -d {input_path} --initializer colmap-dense -o colmap_executable={colmap_location} -oallow_undistortion_missing=1 -ouse_fused=True"
    print("running command:")
    print(command)
    subprocess.check_call(command, shell=True)

def train_initialized(name, input_folder, output_folder, train_indices, iterations, background_folder=None):
    for index in train_indices:
        source_location = os.path.join(input_folder, name, f"frame{index}")
        destination_location = os.path.join(output_folder, name, f"frame{index}")
        if background_folder is None:
            command = f"python -m instantsplat.train -s {source_location} -d {destination_location} -i {iterations} --with_scale_reg "
        else:
            background_location = os.path.join(background_folder, name, "point_cloud/iteration_30000/point_cloud.ply")
            command = f"python -m instantsplat.train -s {source_location} -d {destination_location} -b {background_location} -i {iterations} -ocamera_position_lr_init=0.0 -ocamera_position_lr_final=0.0 -ocamera_rotation_lr_init=0.0 -ocamera_rotation_lr_final=0.0 -ocamera_exposure_lr_init=0.0 -ocamera_exposure_lr_final=0.0 --with_scale_reg"
        print("running command:")
        print(command)
        subprocess.check_call(command, shell=True)

def render(name, output_folder, train_indices, iterations):
    for index in train_indices:
        source_location = os.path.join(output_folder, name, f"frame{index}")
        destincation_location = source_location
        camera_location = os.path.join(source_location, "cameras.json")
        command = f"python -m gaussian_splatting.render -s {source_location} -d {destincation_location} -i {iterations} --load_camera {camera_location} --save_depth_pcd"
        print("running command:")
        print(command)
        subprocess.check_call(command, shell=True)


if __name__ == "__main__":
    align_name = "unaligned_frames3"
    base_folder = "./data"
    background_folder = "./data/backgrounds"
    output_folder = "./output"
    colmap_location = "/home/isaac/miniconda3/envs/gs2/bin/colmap"
    iterations = 1000
    train_indices = range(0, 20)
    train_initialized(align_name, base_folder, output_folder, train_indices, iterations)
    render(align_name, output_folder, train_indices, iterations)