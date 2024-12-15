import sqlite3
import shutil
import os
import re
import subprocess


def execute(cmd):
    proc = subprocess.Popen(cmd, shell=False)
    proc.communicate()
    return proc.returncode


def copy_db(src_database, dst_database):
    conn = sqlite3.connect(dst_database)
    c = conn.cursor()
    c.execute("DELETE FROM cameras")
    c.execute("DELETE FROM images")
    c.execute(f"ATTACH DATABASE '{src_database}' as 'other'")
    c.execute(f"INSERT INTO main.cameras SELECT * FROM other.cameras")
    c.execute(f"INSERT INTO main.images SELECT * FROM other.images")
    conn.commit()
    conn.close()


def model_converter(src_path, dst_path, colmap_executable):
    mapper_input_path = os.path.join(dst_path, "distorted", "sparse", "loading")
    if os.path.isdir(mapper_input_path):
        shutil.rmtree(mapper_input_path)
    os.makedirs(mapper_input_path)
    cmd = [
        colmap_executable, "model_converter",
        "--input_path", os.path.join(src_path, "distorted", "sparse", "0"),
        "--output_path", mapper_input_path,
        "--output_type=TXT",
    ]
    exit_code = execute(cmd)
    if exit_code != 0:
        raise RuntimeError(f"model_converter failed with code {exit_code}. Exiting.")
    with open(mapper_input_path + "/images.txt", "r") as f:
        lines = f.readlines()
    with open(mapper_input_path + "/images.txt", "w") as f:
        for line in lines:
            if line[0] == "#":
                f.writelines([line])
            elif re.match("^[0-9]+ ", line):
                f.writelines([line, "\n"])
    open(mapper_input_path + "/points3D.txt", "w").close()
    return mapper_input_path


def exhaustive_matcher(dst_database, use_gpu, colmap_executable):
    cmd = [
        colmap_executable, "exhaustive_matcher",
        "--database_path", dst_database,
        "--SiftMatching.use_gpu", use_gpu
    ]
    exit_code = execute(cmd)
    if exit_code != 0:
        raise RuntimeError(f"Feature matching failed with code {exit_code}. Exiting.")


def point_triangulator(dst_database, mapper_input_path, image_path, colmap_executable):
    cmd = [
        colmap_executable, "point_triangulator",
        "--database_path", dst_database,
        "--input_path", mapper_input_path,
        "--output_path", mapper_input_path,
        "--image_path", image_path
    ]
    exit_code = execute(cmd)
    if exit_code != 0:
        raise RuntimeError(f"model_converter failed with code {exit_code}. Exiting.")


def load_colmap_cameras(src_path, dst_path, colmap_executable, use_gpu):
    src_database = os.path.join(src_path, "distorted", "database.db")
    dst_database = os.path.join(dst_path, "distorted", "database.db")
    copy_db(src_database, dst_database)
    mapper_input_path = model_converter(src_path, dst_path, colmap_executable)
    # Feature matching
    exhaustive_matcher(dst_database, use_gpu, colmap_executable)
    # Triangulator to get sparse
    point_triangulator(dst_database, mapper_input_path, os.path.join(dst_path, "input"), colmap_executable)
    return mapper_input_path
