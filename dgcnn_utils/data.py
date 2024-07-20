import os
import sys
import glob
import h5py
import numpy as np
import torch
import json
import cv2
import pickle
from torch.utils.data import Dataset


def download_S3DIS():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, "indoor3d_sem_seg_hdf5_data")):
        print("No data avaialable")
        sys.exit(0)
    # if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version')):
    #     if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')):
    #         print('Please download Stanford3dDataset_v1.2_Aligned_Version.zip \
    #             from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under data/')
    #         sys.exit(0)
    # else:
    #     zippath = os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')
    #     os.system('unzip %s' % (zippath))
    #     os.system('mv %s %s' % ('Stanford3dDataset_v1.2_Aligned_Version', DATA_DIR))
    #     os.system('rm %s' % (zippath))


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


def read_txt_file(file_path, num_points: np.int32):
    data_app = []
    labels_app = []
    for f in file_path:
        with open(f) as file:
            count = 0
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


def load_color_partseg():
    colors = []
    labels = []
    f = open("prepare_data/meta/partseg_colors.txt")
    for line in json.load(f):
        colors.append(line["color"])
        labels.append(line["label"])
    partseg_colors = np.array(colors)
    partseg_colors = partseg_colors[:, [2, 1, 0]]
    partseg_labels = np.array(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 1350
    img = np.zeros((1350, 1890, 3), dtype="uint8")
    cv2.rectangle(img, (0, 0), (1900, 1900), [255, 255, 255], thickness=-1)
    column_numbers = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    column_gaps = [320, 320, 300, 300, 285, 285]
    color_size = 64
    color_index = 0
    label_index = 0
    row_index = 16
    for row in range(0, img_size):
        column_index = 32
        for column in range(0, img_size):
            color = partseg_colors[color_index]
            label = partseg_labels[label_index]
            length = len(str(label))
            cv2.rectangle(
                img,
                (column_index, row_index),
                (column_index + color_size, row_index + color_size),
                color=(int(color[0]), int(color[1]), int(color[2])),
                thickness=-1,
            )
            img = cv2.putText(
                img,
                label,
                (
                    column_index + int(color_size * 1.15),
                    row_index + int(color_size / 2),
                ),
                font,
                0.76,
                (0, 0, 0),
                2,
            )
            column_index = column_index + column_gaps[column]
            color_index = color_index + 1
            label_index = label_index + 1
            if color_index >= 50:
                cv2.imwrite(
                    "prepare_data/meta/partseg_colors.png",
                    img,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0],
                )
                return np.array(colors)
            elif column + 1 >= column_numbers[row]:
                break
        row_index = row_index + int(color_size * 1.3)
        if row_index >= img_size:
            break


def load_color_semseg():
    colors = []
    labels = []
    f = open("dgcnn_pytorch/prepare_data/meta/semseg_colors.txt")
    for line in json.load(f):
        colors.append(line["color"])
        labels.append(line["label"])
    semseg_colors = np.array(colors)
    semseg_colors = semseg_colors[:, [2, 1, 0]]
    partseg_labels = np.array(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 1500
    img = np.zeros((500, img_size, 3), dtype="uint8")
    cv2.rectangle(img, (0, 0), (img_size, 750), [255, 255, 255], thickness=-1)
    color_size = 64
    color_index = 0
    label_index = 0
    row_index = 16
    for _ in range(0, img_size):
        column_index = 32
        for _ in range(0, img_size):
            color = semseg_colors[color_index]
            label = partseg_labels[label_index]
            length = len(str(label))
            cv2.rectangle(
                img,
                (column_index, row_index),
                (column_index + color_size, row_index + color_size),
                color=(int(color[0]), int(color[1]), int(color[2])),
                thickness=-1,
            )
            img = cv2.putText(
                img,
                label,
                (
                    column_index + int(color_size * 1.15),
                    row_index + int(color_size / 2),
                ),
                font,
                0.7,
                (0, 0, 0),
                2,
            )
            column_index = column_index + 200
            color_index = color_index + 1
            label_index = label_index + 1
            if color_index >= 13:
                cv2.imwrite(
                    "prepare_data/meta/semseg_colors.png",
                    img,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0],
                )
                return np.array(colors)
            elif column_index >= 1280:
                break
        row_index = row_index + int(color_size * 1.3)
        if row_index >= img_size:
            break


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype(
        "float32"
    )
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.uniform()
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(
        rotation_matrix
    )  # random rotation (x,z)
    return pointcloud


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
