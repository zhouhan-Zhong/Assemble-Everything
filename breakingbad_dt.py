import os
import random

import numpy as np
import tqdm
import trimesh
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset


class GeometryPartDataset(Dataset):
    """Geometry part assembly dataset.

    We follow the data prepared by Breaking Bad dataset:
        https://breaking-bad-dataset.github.io/
    """

    def __init__(
        self,
        data_dir,
        data_fn,
        data_keys,
        category="",
        num_points=1000,
        min_num_part=2,
        max_num_part=2000,
        shuffle_parts=False,
        rot_range=-1,
        overfit=-1,
    ):
        # store parameters
        self.category = category if category.lower() != "all" else ""
        self.data_dir = data_dir
        self.num_points = num_points
        self.min_num_part = min_num_part
        self.max_num_part = max_num_part  # ignore shapes with more parts
        self.shuffle_parts = shuffle_parts  # shuffle part orders
        self.rot_range = rot_range  # rotation range in degree

        # list of fracture folder path
        self.data_list = self._read_data(data_fn)
        if overfit > 0:
            self.data_list = self.data_list[:overfit]

        # additional data to load, e.g. ('part_ids', 'instance_label')
        self.data_keys = data_keys



    def _read_data(self, data_fn):
        """Filter out invalid number of parts."""
        with open(os.path.join(self.data_dir, data_fn), "r") as f:
            mesh_list = [line.strip() for line in f.readlines()]
            if self.category:
                mesh_list = [
                    line for line in mesh_list if self.category in line.split("/")
                ]
        self.used_categories = {
            x.split("/")[1] for x in mesh_list
        }  # get category name from paths, used to initialized the set of metrics

        data_list = []
        print(f"Building dataset")
        for mesh in tqdm.tqdm(mesh_list):
            mesh_dir = os.path.join(self.data_dir, mesh)
            # print(mesh_dir)
            if not os.path.isdir(mesh_dir):
                print(f"{mesh} does not exist")
                continue
            for frac in os.listdir(mesh_dir):
                # we take both fractures and modes for training
                if "fractured" not in frac and "mode" not in frac:
                    continue
                frac = os.path.join(mesh, frac)
                num_parts = len(os.listdir(os.path.join(self.data_dir, frac)))
                print(num_parts)
                if self.min_num_part <= num_parts <= self.max_num_part:
                    data_list.append(frac)
        # print(len(data_list))
        return data_list

    @staticmethod
    def _recenter_pc(pc):
        """pc: [N, 3]"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid[None]
        return pc, centroid

    def _rotate_pc(self, pc):
        """pc: [N, 3]"""
        if self.rot_range > 0.0:
            rot_euler = (np.random.rand(3) - 0.5) * 2.0 * self.rot_range
            rot_mat = R.from_euler("xyz", rot_euler, degrees=True).as_matrix()
        else:
            rot_mat = R.random().as_matrix()
        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        return pc, quat_gt

    @staticmethod
    def _shuffle_pc(pc):
        """pc: [N, 3]"""
        order = np.arange(pc.shape[0])
        random.shuffle(order)
        pc = pc[order]
        return pc

    def _pad_data(self, data):
        """Pad data to shape [`self.max_num_part`, data.shape[1], ...]."""
        data = np.array(data)
        pad_shape = (self.max_num_part,) + tuple(data.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        pad_data[: data.shape[0]] = data
        return pad_data

    def _get_pcs(self, data_folder):
        """Read mesh and sample point cloud from a folder."""
        # `data_folder`: xxx/plate/1d4093ad2dfad9df24be2e4f911ee4af/fractured_0
        data_folder = os.path.join(self.data_dir, data_folder)
        mesh_files = os.listdir(data_folder)
        mesh_files.sort()
        if not self.min_num_part <= len(mesh_files) <= self.max_num_part:
            raise ValueError

        # shuffle part orders
        if self.shuffle_parts:
            random.shuffle(mesh_files)

        # read mesh and sample points
        meshes = [
            trimesh.load(os.path.join(data_folder, mesh_file))
            for mesh_file in mesh_files
        ]
        pcs = [
            trimesh.sample.sample_surface(mesh, self.num_points)[0] for mesh in meshes
        ]
        return np.stack(pcs, axis=0)

    def __getitem__(self, index):
        pcs = self._get_pcs(self.data_list[index])
        num_parts = pcs.shape[0]
        cur_pts, cur_quat, cur_trans = [], [], []
        for i in range(num_parts):
            pc = pcs[i]
            pc, gt_trans = self._recenter_pc(pc)
            pc, gt_quat = self._rotate_pc(pc)
            cur_pts.append(self._shuffle_pc(pc))
            cur_quat.append(gt_quat)
            cur_trans.append(gt_trans)
        cur_pts = self._pad_data(np.stack(cur_pts, axis=0))  # [P, N, 3]
        cur_quat = self._pad_data(np.stack(cur_quat, axis=0))  # [P, 4]
        cur_trans = self._pad_data(np.stack(cur_trans, axis=0))  # [P, 3]
        """
        data_dict = {
            'part_pcs': MAX_NUM x N x 3
                The points sampled from each part.

            'part_trans': MAX_NUM x 3
                Translation vector

            'part_quat': MAX_NUM x 4
                Rotation as quaternion.

            'part_valids': MAX_NUM
                1 for shape parts, 0 for padded zeros.

            'instance_label': MAX_NUM x 0, useless

            'part_label': MAX_NUM x 0, useless

            'part_ids': MAX_NUM, useless

            'data_id': int
                ID of the data.

        }
        """

        data_dict = {
            "part_pcs": cur_pts,
            "part_quat": cur_quat,
            "part_trans": cur_trans,
        }
        # valid part masks
        valids = np.zeros((self.max_num_part), dtype=np.float32)
        valids[:num_parts] = 1.0
        data_dict["part_valids"] = valids
        # data_id
        data_dict["data_id"] = index
        # instance_label is useless in non-semantic assembly
        # keep here for compatibility with semantic assembly
        # make its last dim 0 so that we concat nothing
        instance_label = np.zeros((self.max_num_part, 0), dtype=np.float32)
        data_dict["instance_label"] = instance_label
        # the same goes to part_label
        part_label = np.zeros((self.max_num_part, 0), dtype=np.float32)
        data_dict["part_label"] = part_label

        for key in self.data_keys:
            if key == "part_ids":
                cur_part_ids = np.arange(num_parts)  # p
                data_dict["part_ids"] = self._pad_data(cur_part_ids)

            elif key == "valid_matrix":
                out = np.zeros((self.max_num_part, self.max_num_part), dtype=np.float32)
                out[:num_parts, :num_parts] = 1.0
                data_dict["valid_matrix"] = out

            else:
                raise ValueError(f"ERROR: unknown data {key}")

        return data_dict

    def __len__(self):
        return len(self.data_list)


def build_geometry_dataloader(
    data_dir, 
    data_fn, 
    data_keys, 
    category, 
    num_points, 
    min_num_part, 
    max_num_part, 
    shuffle_parts, 
    rot_range, 
    overfit, 
    batch_size, 
    num_workers
):
    # 创建 data_dict 字典
    data_dict = dict(
        data_dir=data_dir,
        data_fn=data_fn.format("train"),  # 格式化训练数据文件名
        data_keys=data_keys,
        category=category,
        num_points=num_points,
        min_num_part=min_num_part,
        max_num_part=max_num_part,
        shuffle_parts=shuffle_parts,
        rot_range=rot_range,
        overfit=overfit,
    )

    # 创建训练数据集
    train_set = GeometryPartDataset(**data_dict)
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader


if __name__ == "__main__":
    data_dict = dict(
        data_dir=r"D:/frac/breaking_bad/",
        data_fn=r"D:/frac/breaking_bad/data_split/everyday.train.txt",
        data_keys=("part_ids",),
        category="all",  # all
        num_points=1000,
        min_num_part=1,
        max_num_part=200,
        shuffle_parts=False,
        rot_range=-1,
        overfit=-1,
        batch_size=1, 
        num_workers=1
    )

    train_set = build_geometry_dataloader(**data_dict)
    x = train_set
    print(x)
# 假设 dataloader 已经创建好了
for batch_idx, data in enumerate(train_set):
    print(f"Batch {batch_idx}:\n{data}")
    # 如果你想查看数据的形状
    print(f"Shape of batch {batch_idx}: {data.shape}")
    
    # 如果数据包含多个部分（比如图像和标签），你可以分别查看
    # 假设返回的是一个元组 (inputs, labels)
    # inputs, labels = data
    # print(f"Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
    break  # 仅查看第一个批次数据（可以去掉 break 查看所有批次）
