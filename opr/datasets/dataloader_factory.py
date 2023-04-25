"""Functions to create PyTorch DataLoaders for different datasets."""
from typing import Callable, Dict, List, Optional, Tuple, Union

import MinkowskiEngine as ME  # noqa: N817
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader

from opr.datasets.base import BaseDataset
from opr.utils import in_sorted_array


def make_collate_fn(dataset: BaseDataset, batch_split_size: Optional[int] = None) -> Callable:
    """Creates collate_fn function for given dataset.

    Args:
        dataset (BaseDataset): Dataset object.
        batch_split_size (int, optional): Whether to split batches into sub-batches
            for multistaged batch training. Defaults to None.

    Returns:
        Callable: collate_fn function that takes data_list and returns batch.
    """

    def collate_fn(
        data_list: List[Dict[str, Tensor]]
    ) -> Tuple[Union[List[Dict[str, Tensor]], Dict[str, Tensor]], Tensor, Tensor]:
        """Pack input data list into batch.

        Args:
            data_list (List[Dict[str, Tensor]]]): batch data list
                generated by DataLoader.

        Raises:
            NotImplementedError: If trying to use multistaged training.

        Returns:
            Dict[str, Tensor]: dictionary of batched data.
        """
        if "cloud" in data_list[0]:
            clouds: Union[Tensor, List[Tensor]] = [e["cloud"] for e in data_list]
            n_points = [int(e.shape[0]) for e in clouds]
            clouds = torch.cat(list(clouds), dim=0).unsqueeze(0)  # (1, batch_size*n_points, 3) tensor
            if dataset.cloud_set_transform is not None:
                # Apply the same transformation on all dataset elements
                clouds = dataset.cloud_set_transform(clouds)
            clouds = torch.split(clouds.squeeze(0), split_size_or_sections=n_points, dim=0)  # back to list
            quantized_coords = [
                ME.utils.sparse_quantize(coordinates=e, quantization_size=dataset.mink_quantization_size)
                for e in clouds
            ]
        if "image" in data_list[0]:
            images = [e["image"] for e in data_list]
        if "range_image" in data_list[0]:
            range_images = [e["range_image"] for e in data_list]

        utms = torch.stack([e["utm"] for e in data_list], dim=0)

        result: Union[List[Dict[str, Tensor]], Dict[str, Tensor]]

        if batch_split_size is None or batch_split_size == 0:
            result = {}
            if "cloud" in data_list[0]:
                result["coordinates"] = ME.utils.batched_coordinates(quantized_coords)
                result["features"] = torch.ones((result["coordinates"].shape[0], 1), dtype=torch.float32)
            if "image" in data_list[0]:
                result["images"] = torch.stack(images, dim=0)
            if "range_image" in data_list[0]:
                result["range_images"] = torch.stack(range_images, dim=0)
            result["utms"] = utms
        else:  # split the batch into chunks
            raise NotImplementedError("Multistaged batch training not yet implemented")

        indices = [int(e["idx"]) for e in data_list]
        positives_mask_list = [
            [in_sorted_array(e, dataset.get_positives(label)) for e in indices] for label in indices
        ]
        negatives_mask_list = [
            [not in_sorted_array(e, dataset.get_nonnegatives(label)) for e in indices] for label in indices
        ]
        positives_mask = torch.tensor(positives_mask_list)
        negatives_mask = torch.tensor(negatives_mask_list)
        return result, positives_mask, negatives_mask

    return collate_fn


def make_dataloaders(
    dataset_cfg: DictConfig,
    batch_sampler_cfg: DictConfig,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Function to create DataLoader objects from given dataset and sampler configs.

    Args:
        dataset_cfg (DictConfig): Dataset configuration.
        batch_sampler_cfg (DictConfig): Batch sampler configuration.
        num_workers (int): Number of workers for DataLoader. Defaults to 0.

    Returns:
        Dict[str, DataLoader]: Dictionary with DataLoaders.
    """
    dataset = {}
    for subset in ["train", "val", "test"]:
        dataset[subset] = instantiate(dataset_cfg, subset=subset)

    batch_split_size: Dict[str, Optional[int]] = {}
    if "batch_split_size" not in batch_sampler_cfg:
        batch_split_size["train"] = None
        batch_split_size["val"] = None
    else:
        batch_split_size["train"] = batch_sampler_cfg.batch_split_size
        batch_split_size["val"] = batch_sampler_cfg.batch_split_size

    sampler = {}
    sampler["train"] = instantiate(batch_sampler_cfg, dataset=dataset["train"])
    if "val_batch_size" in batch_sampler_cfg and batch_sampler_cfg.val_batch_size is not None:
        val_batch_size = batch_sampler_cfg.val_batch_size
        sampler["val"] = instantiate(
            batch_sampler_cfg,
            dataset=dataset["val"],
            batch_size=val_batch_size,
            batch_size_limit=None,
            batch_expansion_rate=None,
        )
        batch_split_size["val"] = None
    elif "batch_size_limit" not in batch_sampler_cfg or batch_sampler_cfg.batch_size_limit is None:
        val_batch_size = batch_sampler_cfg.batch_size
        sampler["val"] = instantiate(batch_sampler_cfg, dataset=dataset["val"])
    else:
        val_batch_size = batch_sampler_cfg.batch_size_limit
        sampler["val"] = instantiate(
            batch_sampler_cfg,
            dataset=dataset["val"],
            batch_size=val_batch_size,
            batch_size_limit=None,
            batch_expansion_rate=None,
        )

    dataloaders = {}
    for subset in ["train", "val"]:
        dataloaders[subset] = DataLoader(
            dataset=dataset[subset],
            batch_sampler=sampler[subset],
            collate_fn=make_collate_fn(dataset[subset], batch_split_size=batch_split_size[subset]),
            num_workers=num_workers,
            pin_memory=True,
        )
    dataloaders["test"] = DataLoader(
        dataset=dataset["test"],
        batch_size=val_batch_size,
        collate_fn=make_collate_fn(dataset["test"], batch_split_size=None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataloaders