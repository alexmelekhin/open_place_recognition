"""Custom ITLP-Campus dataset implementations."""
import math
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import cv2
import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset
import MinkowskiEngine as ME

from opr.datasets.augmentations import (
    DefaultCloudSetTransform,
    DefaultCloudTransform,
    DefaultImageTransform,
    DefaultSemanticTransform,
)


class ITLPCampus(Dataset):
    """ITLP Campus dataset implementation."""

    dataset_root: Path
    dataset_df: DataFrame
    front_cam_text_descriptions_df: Optional[DataFrame]
    back_cam_text_descriptions_df: Optional[DataFrame]
    front_cam_text_labels_df: Optional[DataFrame]
    back_cam_text_labels_df: Optional[DataFrame]
    front_cam_aruco_labels_df: Optional[DataFrame]
    back_cam_aruco_labels_df: Optional[DataFrame]
    sensors: Tuple[str, ...]
    images_subdir: str = ""
    clouds_subdir: str = "lidar"
    semantic_subdir: str = "masks"
    text_descriptions_subdir: str = "text_descriptions"
    text_labels_subdir: str = "text_labels"
    aruco_labels_subdir: str = "aruco_labels"
    image_transform: DefaultImageTransform
    pointcloud_transform: DefaultCloudTransform
    cloud_set_transform: DefaultCloudSetTransform
    _pointcloud_quantization_size: Optional[float]
    load_semantics: bool
    load_text_descriptions: bool
    load_text_labels: bool
    load_aruco_labels: bool
    indoor: bool

    def __init__(
        self,
        dataset_root: Union[str, Path],
        csv_file: str = "track.csv",
        sensors: Union[str, Tuple[str, ...]] = ("front_cam", "lidar"),
        mink_quantization_size: Optional[float] = 0.5,
        max_point_distance: Optional[float] = None,
        load_semantics: bool = False,
        exclude_dynamic_classes: bool = False,
        load_text_descriptions: bool = False,
        load_text_labels: bool = False,
        load_aruco_labels: bool = False,
        indoor: bool = False,
        positive_threshold: float = 10.0,
        negative_threshold: float = 50.0,
        image_transform = DefaultImageTransform(resize=(320, 192), train=False),
        semantic_transform = DefaultSemanticTransform(resize=(320, 192), train=False)
    ) -> None:
        """ITLP Campus dataset implementation.

        Args:
            dataset_root (Union[str, Path]): Path to the dataset track root directory.
            csv_file (str): Name of the csv file with dataset information. Defaults to "track.csv".
            sensors (Union[str, Tuple[str, ...]]): List of sensors for which the data should be loaded.
                Defaults to ("front_cam", "lidar").
            mink_quantization_size (Optional[float]): The quantization size for point clouds. Defaults to 0.5.
            load_semantics (bool): Wether to load semantic masks for camera images. Defaults to False.
            load_text_descriptions (bool): Wether to load text descriptions for camera images.
                Defaults to False.
            load_text_labels (bool): Wether to load detected text for camera images. Defaults to False.
            load_aruco_labels (bool): Wether to load detected aruco labels for camera images.
                Defaults to False.
            indoor (bool): Wether to load indoor or outdoor dataset track. Defaults to False.

        Raises:
            FileNotFoundError: If dataset_root doesn't exist.
            FileNotFoundError: If there is no csv file for given subset (track).
        """
        super().__init__()

        self.dataset_root = Path(dataset_root)
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Given dataset_root={self.dataset_root} doesn't exist")

        subset_csv = self.dataset_root / csv_file
        self.dataset_df = pd.read_csv(subset_csv)

        if isinstance(sensors, str):
            sensors = tuple([sensors])
        self.sensors = sensors

        self._pointcloud_quantization_size = mink_quantization_size
        self._max_point_distance = max_point_distance
        self.load_semantics = load_semantics

        self.load_text_descriptions = load_text_descriptions
        if self.load_text_descriptions:
            if "front_cam" in self.sensors:
                self.front_cam_text_descriptions_df = pd.read_csv(
                    self.dataset_root / self.text_descriptions_subdir / "front_cam_text.csv"
                )
            if "back_cam" in self.sensors:
                self.back_cam_text_descriptions_df = pd.read_csv(
                    self.dataset_root / self.text_descriptions_subdir / "back_cam_text.csv"
                )

        self.load_text_labels = load_text_labels
        if self.load_text_labels:
            if "front_cam" in self.sensors:
                self.front_cam_text_labels_df = pd.read_csv(
                    self.dataset_root / self.text_labels_subdir / "front_cam_text_labels.csv"
                )
            if "back_cam" in self.sensors:
                self.back_cam_text_labels_df = pd.read_csv(
                    self.dataset_root / self.text_labels_subdir / "back_cam_text_labels.csv"
                )

        self.load_aruco_labels = load_aruco_labels
        if self.load_aruco_labels:
            if "front_cam" in self.sensors:
                self.front_cam_aruco_labels_df = pd.read_csv(
                    self.dataset_root / self.aruco_labels_subdir / "front_cam_aruco_labels.csv", sep="\t"
                )
            if "back_cam" in self.sensors:
                self.back_cam_aruco_labels_df = pd.read_csv(
                    self.dataset_root / self.aruco_labels_subdir / "back_cam_aruco_labels.csv", sep="\t"
                )

        self.indoor = indoor

        # omg so wet 💦💦💦
        if positive_threshold < 0.0:
            raise ValueError(f"positive_threshold must be non-negative, but {positive_threshold!r} given.")
        if negative_threshold < 0.0:
            raise ValueError(f"negative_threshold must be non-negative, but {negative_threshold!r} given.")

        self._positives_index, self._nonnegative_index = self._build_indexes(
            positive_threshold, negative_threshold
        )
        self._positives_mask, self._negatives_mask = self._build_masks(positive_threshold, negative_threshold)

        self.image_transform = image_transform
        self.semantic_transform = semantic_transform
        self.pointcloud_transform = DefaultCloudTransform(train=False)
        self.pointcloud_set_transform = DefaultCloudSetTransform(train=False)

        self._ade20k_dynamic_idx = [12]
        self.exclude_dynamic_classes = exclude_dynamic_classes

        self.lidar2front = np.array([[ 0.01509615, -0.99976457, -0.01558544,  0.04632156],
                                    [ 0.00871086,  0.01571812, -0.99983852, -0.13278588],
                                    [ 0.9998481,   0.01495794,  0.0089461,  -0.06092749],
                                    [ 0.      ,    0.  ,        0.    ,      1.        ]])
        self.lidar2back = np.array([[-1.50409674e-02,  9.99886421e-01,  9.55906151e-04,  1.82703304e-02],
                                    [-1.30440106e-02,  7.59716299e-04, -9.99914635e-01, -1.41787545e-01],
                                    [-9.99801792e-01, -1.50521522e-02,  1.30311022e-02, -6.72336358e-02],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        
        self.front_matrix = np.array([[683.6199340820312, 0.0, 615.1160278320312, 0.0, 683.6199340820312, 345.32354736328125, 0.0, 0.0, 1.0]]).reshape((3,3))
        self.front_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.back_matrix = np.array([[910.4178466796875, 0.0, 648.44140625, 0.0, 910.4166870117188, 354.0118408203125, 0.0, 0.0, 1.0]]).reshape((3,3))
        self.back_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    def __getitem__(self, idx: int) -> Dict[str, Union[int, Tensor]]:  # noqa: D105
        data: Dict[str, Union[int, Tensor]] = {"idx": torch.tensor(idx)}
        row = self.dataset_df.iloc[idx]
        data["pose"] = torch.tensor(
            self.dataset_df.iloc[idx][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float32)
        )
        floor = self._get_floor_subdir(idx)
        track = self._get_track_subdir(idx)
        if "front_cam" in self.sensors:
            image_ts = int(self.dataset_df["front_cam_ts"].iloc[idx])
            im_filepath = self.dataset_root / track / floor / self.images_subdir / "front_cam" / f"{image_ts}.png"
            im = cv2.imread(str(im_filepath))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = self.image_transform(im)
            data["image_front_cam"] = im
            if self.load_semantics:
                im_filepath = (
                    self.dataset_root / track / floor / self.semantic_subdir / "front_cam" / f"{image_ts}.png"
                )  # image id is equal to semantic mask id~
                im = cv2.imread(str(im_filepath), cv2.IMREAD_UNCHANGED)
                im = self.semantic_transform(im)
                data["mask_front_cam"] = im

                if self.exclude_dynamic_classes and self.indoor:
                    for index in self._ade20k_dynamic_idx:
                        data["image_front_cam"] = torch.where(data["mask_front_cam"] == index, 0, data["image_front_cam"])

            if self.load_text_labels:
                text_labels_df = self.front_cam_text_labels_df[
                    self.front_cam_text_labels_df["path"] == f"{image_ts}.png"
                ]
                data["text_labels_front_cam_df"] = text_labels_df
            if self.load_text_descriptions:
                text_description_df = self.front_cam_text_descriptions_df[
                    self.front_cam_text_descriptions_df["path"] == f"{image_ts}.png"
                ]
                data["text_description_front_cam_df"] = text_description_df
            if self.load_aruco_labels:
                aruco_labels_df = self.front_cam_aruco_labels_df[
                    self.front_cam_aruco_labels_df["image_name"] == f"{image_ts}.png"
                ]
                data["aruco_labels_front_cam_df"] = aruco_labels_df
        if "back_cam" in self.sensors:
            image_ts = int(self.dataset_df["back_cam_ts"].iloc[idx])
            im_filepath = self.dataset_root / track / floor / self.images_subdir / "back_cam" / f"{image_ts}.png"
            im = cv2.imread(str(im_filepath))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = self.image_transform(im)
            data["image_back_cam"] = im
            if self.load_semantics:
                im_filepath = (
                    self.dataset_root / track / floor / self.semantic_subdir / "back_cam" / f"{image_ts}.png"
                )  # image id is equal to semantic mask id~
                im = cv2.imread(str(im_filepath), cv2.IMREAD_UNCHANGED)
                im = self.semantic_transform(im)
                data["mask_back_cam"] = im

                if self.exclude_dynamic_classes and self.indoor:
                    for index in self._ade20k_dynamic_idx:
                        data["image_back_cam"] = torch.where(data["mask_back_cam"] == index, 0, data["image_back_cam"])

            if self.load_text_labels:
                text_labels_df = self.back_cam_text_labels_df[
                    self.back_cam_text_labels_df["path"] == f"{image_ts}.png"
                ]
                data["text_labels_back_cam_df"] = text_labels_df
            if self.load_text_descriptions:
                text_description_df = self.back_cam_text_descriptions_df[
                    self.back_cam_text_descriptions_df["path"] == f"{image_ts}.png"
                ]
                data["text_description_back_cam_df"] = text_description_df
            if self.load_aruco_labels:
                aruco_labels_df = self.back_cam_aruco_labels_df[
                    self.back_cam_aruco_labels_df["image_name"] == f"{image_ts}.png"
                ]
                data["aruco_labels_back_cam_df"] = aruco_labels_df
        if "lidar" in self.sensors:
            lidar_ts = int(self.dataset_df["lidar_ts"].iloc[idx])
            pc_filepath = self.dataset_root / track / floor / self.clouds_subdir / f"{lidar_ts}.bin"
            pc = self._load_pc(pc_filepath)

            if self.exclude_dynamic_classes and self.indoor:
                if "back_cam" in self.sensors:
                    pc = self._remove_dynamic_points(pc, data["mask_back_cam"].numpy().transpose(1, 2, 0),
                                                     self.lidar2back, self.back_matrix, self.back_dist)
                if "front_cam" in self.sensors:
                    pc = self._remove_dynamic_points(pc, data["mask_front_cam"].numpy().transpose(1, 2, 0),
                                                     self.lidar2front, self.front_matrix, self.front_dist)

            pc = torch.tensor(pc, dtype=torch.float32)
            data["pointcloud_lidar_coords"] = pc
            data["pointcloud_lidar_feats"] = torch.ones_like(pc[:, :1])
        return data
    
    def _remove_dynamic_points(self, pointcloud: np.ndarray, semantic_map: np.ndarray, lidar2sensor: np.ndarray,
                               sensor_intrinsics: np.ndarray, sensor_dist: np.ndarray) -> np.ndarray:
        pc_values = np.concatenate([pointcloud, np.ones((pointcloud.shape[0], 1))],axis=1).T
        camera_values = lidar2sensor @ pc_values
        camera_values = np.transpose(camera_values)[:, :3]

        points_2d, _ = cv2.projectPoints(camera_values, 
                                         np.zeros((3, 1), np.float32), np.zeros((3, 1), np.float32), 
                                         sensor_intrinsics, 
                                         sensor_dist)
        points_2d = points_2d[:, 0, :]

        classes = set(np.unique(semantic_map))
        dynamic_classes = set(self._ade20k_dynamic_idx)
        if classes.intersection(dynamic_classes):
            valid = (~np.isnan(points_2d[:,0])) & (~np.isnan(points_2d[:,1]))
            in_bounds_x = (points_2d[:,0] >= 0) & (points_2d[:,0] < 1280)
            in_bounds_y = (points_2d[:,1] >= 0) & (points_2d[:,1] < 720)
            look_forward = (camera_values[:, 2] > 0)
            mask = valid & in_bounds_x & in_bounds_y & look_forward

            indices = np.where(mask)[0]
            mask_for_points = np.full((points_2d.shape[0], 3), True)

            dynamic_idx = np.array(self._ade20k_dynamic_idx)
            semantic_values = semantic_map[np.floor(points_2d[indices, 1]).astype(int), np.floor(points_2d[indices, 0]).astype(int)]

            matching_indices = np.where(np.isin(semantic_values, dynamic_idx))

            mask_for_points = np.full((points_2d.shape[0], 3), True)
            mask_for_points[indices[matching_indices[0]]] = np.array([False, False, False])

            return pointcloud[mask_for_points].reshape((-1, 3))
        else:
            return pointcloud

    def __len__(self) -> int:  # noqa: D105
        return len(self.dataset_df)

    def _get_floor_subdir(self, idx: int) -> str:
        if "floor" in self.dataset_df.columns:
            return f"floor_{self.dataset_df['floor'].iloc[idx]}"
        else:
            return ""

    def _get_track_subdir(self, idx: int) -> str:
        if "track" in self.dataset_df.columns:
            return self.dataset_df['track'].iloc[idx]
        else:
            return ""

    def _load_pc(self, filepath: Union[str, Path]) -> Tensor:
        pc = np.fromfile(filepath, dtype=np.float32).reshape((-1, 4))[:, :-1]
        in_range_idx = np.all(
            np.logical_and(-100 <= pc, pc <= 100),  # select points in range [-100, 100] meters
            axis=1,
        )
        pc = pc[in_range_idx]
        if self._max_point_distance is not None:
            pc = pc[np.linalg.norm(pc, axis=1) < self._max_point_distance]
        return pc

    def _collate_data_dict(self, data_list: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        result: Dict[str, Tensor] = {}
        result["idxs"] = torch.stack([e["idx"] for e in data_list], dim=0)
        for data_key in data_list[0].keys():
            if data_key == "idx":
                continue
            elif data_key == "pose":
                result["poses"] = torch.stack([e["pose"] for e in data_list], dim=0)
            elif data_key.startswith("image_"):
                result[f"images_{data_key[6:]}"] = torch.stack([e[data_key] for e in data_list])
            elif data_key.startswith("mask_"):
                result[f"masks_{data_key[5:]}"] = torch.stack([e[data_key] for e in data_list])
            elif data_key == "pointcloud_lidar_coords":
                coords_list = [e["pointcloud_lidar_coords"] for e in data_list]
                feats_list = [e["pointcloud_lidar_feats"] for e in data_list]
                n_points = [int(e.shape[0]) for e in coords_list]
                coords_tensor = torch.cat(coords_list, dim=0).unsqueeze(0)  # (1,batch_size*n_points,3)
                if self.pointcloud_set_transform is not None:
                    # Apply the same transformation on all dataset elements
                    coords_tensor = self.pointcloud_set_transform(coords_tensor)
                coords_list = torch.split(coords_tensor.squeeze(0), split_size_or_sections=n_points, dim=0)
                quantized_coords_list = []
                quantized_feats_list = []
                for coords, feats in zip(coords_list, feats_list):
                    quantized_coords, quantized_feats = ME.utils.sparse_quantize(
                        coordinates=coords,
                        features=feats,
                        quantization_size=self._pointcloud_quantization_size,
                    )
                    quantized_coords_list.append(quantized_coords)
                    quantized_feats_list.append(quantized_feats)

                result["pointclouds_lidar_coords"] = ME.utils.batched_coordinates(quantized_coords_list)
                result["pointclouds_lidar_feats"] = torch.cat(quantized_feats_list)
            elif data_key == "pointcloud_lidar_feats":
                continue
            else:
                raise ValueError(f"Unknown data key: {data_key!r}")
        return result

    def collate_fn(self, data_list: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """Pack input data list into batch.

        Args:
            data_list (List[Dict[str, Tensor]]): batch data list generated by DataLoader.

        Returns:
            Dict[str, Tensor]: dictionary of batched data.
        """
        return self._collate_data_dict(data_list)

    # omg so wet 💦💦💦
    def _build_masks(self, positive_threshold: float, negative_threshold: float) -> Tuple[Tensor, Tensor]:
        """Build boolean masks for dataset elements that satisfy a UTM distance threshold condition.

        Args:
            positive_threshold (float): The maximum UTM distance between two elements
                for them to be considered positive.
            negative_threshold (float): The maximum UTM distance between two elements
                for them to be considered non-negative.

        Returns:
            Tuple[Tensor, Tensor]: A tuple of two boolean masks that satisfy the UTM distance threshold
                condition for each element in the dataset. The first mask contains the indices of elements
                that satisfy the positive threshold, while the second mask contains the indices of elements
                that satisfy the negative threshold.
        """
        xyz = torch.tensor(
            self.dataset_df[["tx", "ty", "tz"]].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        distances = torch.cdist(xyz, xyz)

        positives_mask = (distances > 0) & (distances < positive_threshold)
        negatives_mask = distances > negative_threshold

        return positives_mask, negatives_mask

    def _build_indexes(
        self, positive_threshold: float, negative_threshold: float
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Build index of elements that satisfy a UTM distance threshold condition.

        Args:
            positive_threshold (float): The maximum UTM distance between two elements
                for them to be considered positive.
            negative_threshold (float): The maximum UTM distance between two elements
                for them to be considered non-negative.

        Returns:
            Tuple[List[Tensor], List[Tensor]]: Tuple (positive_indices, nonnegative_indices)
                of two lists of element indexes that satisfy the UTM distance threshold condition
                for each element in the dataset.
        """
        xyz = torch.tensor(
            self.dataset_df[["tx", "ty", "tz"]].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        distances = torch.cdist(xyz, xyz)

        positives_mask = (distances > 0) & (distances < positive_threshold)
        nonnegatives_mask = distances < negative_threshold

        # Convert the boolean masks to index tensors
        positive_indices = [torch.nonzero(row).squeeze(dim=-1) for row in positives_mask]
        nonnegative_indices = [torch.nonzero(row).squeeze(dim=-1) for row in nonnegatives_mask]

        return positive_indices, nonnegative_indices

    @property
    def positives_index(self) -> List[Tensor]:
        """List of indexes of positive samples for each element in the dataset."""
        return self._positives_index

    @property
    def nonnegative_index(self) -> List[Tensor]:
        """List of indexes of non-negatives samples for each element in the dataset."""
        return self._nonnegative_index

    @property
    def positives_mask(self) -> Tensor:
        """Boolean mask of positive samples for each element in the dataset."""
        return self._positives_mask

    @property
    def negatives_mask(self) -> Tensor:
        """Boolean mask of negative samples for each element in the dataset."""
        return self._negatives_mask

    @staticmethod
    def download_data(out_dir: Union[Path, str]) -> None:
        outdoor_tracks_dict = {
            "00_2023-02-10": "17HVoPmM7iR1f2Aj8H9GYzOqieCKwjh96",
            "01_2023-02-21": "1mezN1c8-3ylZrub9_lnGlJzipr90K63O",
            "02_2023-03-15": "1lKdW7ZfpaNLiIQtoJozoSqx397H7iwb1",
            "03_2023-04-11": "18t79U4IKxABTMYdSBOafwlUGlYvJcltx",
            "04_2023-04-13": "1KMTMU-oxXbBV8bmtAY1g8GsquGFksDcE",
        }
        indoor_tracks_dict = {
            "00_2023-03-13": "1AFPKdMrXwPlcC50d1Y8DL4g11CbD31Q2",
        }

        out_dir = Path(out_dir)
        if not out_dir.exists():
            print(f"Creating output directory: {out_dir}")
            out_dir.mkdir(parents=True)
        else:
            print(f"Will download in existing directory: {out_dir}")

        outdoor_dir = out_dir / "ITLP_Campus_outdoor"
        outdoor_dir.mkdir(exist_ok=True)
        for track_name, file_id in outdoor_tracks_dict.items():
            gdown.download(
                f"https://drive.google.com/uc?export=download&confirm=pbef&id={file_id}",
                output=str(outdoor_dir / f"{track_name}.zip"),
                quiet=False,
                fuzzy=False,
                use_cookies=False,
            )
        indoor_dir = out_dir / "ITLP_Campus_indoor"
        indoor_dir.mkdir(exist_ok=True)
        for track_name, file_id in indoor_tracks_dict.items():
            gdown.download(
                f"https://drive.google.com/uc?export=download&confirm=pbef&id={file_id}",
                output=str(indoor_dir / f"{track_name}.zip"),
                quiet=False,
                fuzzy=False,
                use_cookies=False,
            )
