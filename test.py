import os
import subprocess

def initialize(name, folder, clean = True, max_search_frames = 255):
    colmap_location = "\'/home/isaac/miniconda3/envs/gs2/bin/colmap\'"
    input_path = os.path.join(folder, name)
    if clean:
        import shutil
        paths= []
        paths.append(os.path.join(input_path, "background"))
        paths.append(os.path.join(input_path, "distorted"))
        paths.append(os.path.join(input_path, "images"))
        paths.append(os.path.join(input_path, "sparse"))
        paths.append(os.path.join(input_path, "stereo"))
        paths.append(os.path.join(input_path, "point_cloud"))
        paths.append(os.path.join(input_path, "fused.ply"))
        paths.append(os.path.join(input_path, "fused.ply.vis"))
        paths.append(os.path.join(input_path, "input.ply"))
        paths.append(os.path.join(input_path, "run-colmap-geometric.sh"))
        paths.append(os.path.join(input_path, "run-colmap-photometric.sh"))
        paths.append(os.path.join(input_path, "cameras.json"))
        paths.append(os.path.join(input_path, "cfg_args"))

        for search_idx in range(max_search_frames):
            search_path = os.path.join(input_path, f"frame{search_idx}")
            paths.append(search_path)
        
        for path in paths:
            if os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)

    command = f"python ./initialize.py -d {input_path} --initializer colmap-dense -oallow_undistortion_missing=1 -ouse_fused=True"
    print("running command:")
    print(command)
    subprocess.check_call(command, shell=True)

def reinitialize(name, data_folder, background_folder):
    input_path = os.path.join(data_folder, name)
    background_path = os.path.join(background_folder, name, "cameras.json")
    command = f"python ./initialize.py -d {input_path} --initalizer colmap-dense -oallow_undistortion_missing=1 -ouse_fused=True --load_camera {background_path}"
    print("running command:")
    print(command)
    subprocess.check_call(command, shell=True)

def train_background(name, data_folder, background_folder, iterations):
    input_path = os.path.join(data_folder, name)
    output_path = os.path.join(background_folder, name)
    command = f"python -m reduced_3dgs.train -s {input_path} -d {output_path} -i {iterations} --mode camera-densify-prune-shculling --with_scale_reg --empty_cache_every_step --no_depth_data"
    print("running command:")
    print(command)
    subprocess.check_call(command, shell=True)

def get_train_indices(name, data_folder, max_window = 255):
    train_path = os.path.join(data_folder, name)
    found_indices = []
    for i in range(max_window):
        train_folder = os.path.join(train_path, f"frame{i}")
        if not os.path.exists(train_folder):
            continue
        files = os.listdir(os.path.join(train_folder, "images"))
        if len(files) < 3:
            continue
        found_indices.append(i)
    return found_indices

def train_initialized(name, input_folder, output_folder, train_indices, iterations, background_folder=None, background_iterations=1000):
    for index in train_indices:
        source_location = os.path.join(input_folder, name, f"frame{index}")
        destination_location = os.path.join(output_folder, name, f"frame{index}")
        if background_folder is None:
            command = f"python -m instantsplat.train -s {source_location} -d {destination_location} -i {iterations} --with_scale_reg "
        else:
            background_location = os.path.join(background_folder, name, f"point_cloud/iteration_{background_iterations}/point_cloud.ply")
            command = f"python -m instantsplat.train -s {source_location} -d {destination_location} -i {iterations} --with_scale_reg -ocamera_position_lr_init=0 -ocamera_position_lr_final=0 -ocamera_rotation_lr_init=0 -ocamera_rotation_lr_final=0 --load_background {background_location}"
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

def complete_process(name, data_folder, output_folder, background_folder, foreground_iterations = 1000, background_iterations = 1000):
    initialize(name, data_folder)
    train_background(name, data_folder, background_folder, background_iterations)
    indices = get_train_indices(name, data_folder)
    train_initialized(name, data_folder, output_folder, indices, foreground_iterations, background_folder, background_iterations)
    render(name, output_folder, indices, foreground_iterations)

if __name__ == "__main__":
    align_name = "aligned_frames5"
    data_folder = "./data"
    output_folder = "./output"
    complete_process(name = align_name,
                     data_folder = data_folder,
                     output_folder = output_folder,
                     background_folder = os.path.join(output_folder, "backgrounds"),
                     foreground_iterations = 1000,
                     background_iterations = 30000)