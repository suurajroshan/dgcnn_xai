import os
import sys
import glob
import h5py
import numpy as np
import torch
import json
import cv2
from torch.utils.data import Dataset


def download_S3DIS():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, "indoor3d_sem_seg_hdf5_data")):
        print("No data avaialable")
        sys.exit(0)


def prepare_test_data_semseg():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    if not os.path.exists(os.path.join(DATA_DIR, "stanford_indoor3d")):
        os.system("python prepare_data/collect_indoor3d_data.py")
    if not os.path.exists(os.path.join(DATA_DIR, "indoor3d_sem_seg_hdf5_data_test")):
        os.system("python prepare_data/gen_indoor3d_h5.py")


def load_data_semseg(partition, test_area):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    download_S3DIS()  # downloads both "Stanford3dDataset_v1.2_Aligned_Version" & "indoor3d_sem_seg_hdf5_data"
    # prepare_test_data_semseg()
    if partition == "train":
        data_dir = os.path.join(DATA_DIR, "indoor3d_sem_seg_hdf5_data")
    else:
        data_dir = os.path.join(DATA_DIR, "indoor3d_sem_seg_hdf5_data_test")
    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files:
        file = h5py.File(os.path.join(DATA_DIR, f), "r+")
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)
    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    test_area_name = "Area_" + test_area
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    if partition == "train":
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg


def read_txt_file(file_path):
    data_app = []
    labels_app = []
    for f in file_path:
        with open(f) as file:
            data = []
            labels = []
            for line in file:
                values = line.strip().split(" ")
                data.append([np.float32(value) for value in values[:-1]])
                labels.append(np.int32(float(values[-1])))
            data = np.array(data)
            labels = np.array(labels)
            data_app = data if len(data_app) == 0 else np.dstack((data_app, data))
            labels_app = (
                labels if len(labels_app) == 0 else np.dstack((labels_app, labels))
            )
    return np.transpose(data_app, (2, 0, 1)), np.squeeze(np.transpose(labels_app))


def get_data_files(data_path, num_points: int):
    output_file = "all_pts_cloud_data.h5"

    str_fname = [
        line.rstrip()
        for line in open(os.path.join(data_path, "synsetoffset2category.txt"))
    ][0]
    sub_filename = str_fname.split()[1]

    json_files = glob.glob(os.path.join(data_path, "train_test_split", "*.json"))
    if os.path.isfile(os.path.join(data_path, output_file)) is False:
        for file in json_files:
            fname = file.split("/")[-1]
            stage_name = fname.split("_")[1]
            f = open(file)
            jsonf = json.load(f)
            file_arr = [line.rstrip().split("/")[-1] for line in jsonf]
            file_paths = [
                os.path.join(data_path, sub_filename, i + ".txt") for i in file_arr
            ]
            out_data, out_labels = read_txt_file(
                file_path=file_paths, num_points=num_points
            )

            with h5py.File(os.path.join(data_path, output_file), "a") as hfile:
                group = hfile.create_group(str(stage_name))
                group.create_dataset("data", data=out_data)
                group.create_dataset("labels", data=out_labels)
                print("%s data group created" % str(stage_name))


def load_data_file(data_path, stage_name):
    file = os.path.join(data_path, "all_pts_cloud_data.h5")
    f = h5py.File(file, "r")
    data = f[str(stage_name)]["data"][:]
    label = f[str(stage_name)]["labels"][:]
    return (data, label)


class S3DIS(Dataset):
    def __init__(self, num_points=2048, partition="train", test_area="1"):
        self.data, self.seg = load_data_semseg(partition, test_area)
        self.data = self.data[:, :, :6]  # use [x y z nx ny nz] features
        print(self.data.shape, "data shape")
        print(self.seg.shape, "seg shape")
        self.num_points = num_points
        self.partition = partition
        # self.semseg_colors = load_color_semseg()

    def __getitem__(self, item):
        pointcloud = self.data[item][: self.num_points]
        seg = self.seg[item][: self.num_points]
        if self.partition == "train":
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        seg = torch.LongTensor(seg)
        return pointcloud, seg

    def __len__(self):
        return self.data.shape[0]


class our_data(Dataset):
    def __init__(self, num_points=2048, partition="train", data_path=None):
        get_data_files(data_path, num_points)
        print(".h5 file saved")
        self.data, self.seg = load_data_file(data_path, partition)
        self.data = self.data[:, :, :6]  # use [x y z nx ny nz] features
        self.num_points = num_points
        self.partition = partition
        print("Original CLASSES: ", np.unique(self.seg))

        class_map = {1: 0, 2: 1, 3: 2, 4: 3, 7: 4}
        self.seg = np.vectorize(class_map.get)(self.seg)
        print("Remapped CLASSES: ", np.unique(self.seg))

    def __getitem__(self, item):
        pointcloud = self.data[item][: self.num_points]
        seg = self.seg[item][: self.num_points]
        if self.partition == "train":
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        seg = torch.LongTensor(seg)
        return pointcloud, seg

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    train = S3DIS(4096)
    test = S3DIS(4096, "test")
    data, seg = train[0]
    print(data.shape)
    print(seg.shape)
