
import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.hero_dataset import HERODataset

@registry.register_builder("hero_dataset")
class HEROBuilder(BaseDatasetBuilder):
    train_dataset_cls = HERODataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/hero_dataset.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        dataset_cls = self.train_dataset_cls
        
        # Get config
        ann_path = build_info.get("ann_path", None)
        vis_root = build_info.get("vis_root", None) # Or build_info.image_path?
        # Standard configs often use 'storage' under 'images' for vis_root.
        # But let's check config schema.
        # In base_dataset_builder, it usually expects build_info.storage if specific data_type.
        
        # Let's support direct keys if possible, or follow standard.
        # mer2024 uses build_info.image_path.
        
        if vis_root is None:
             vis_root = build_info.get("image_path", build_info.get("storage", ""))
             
        modality_dropout_prob = build_info.get("modality_dropout_prob", 0.0)

        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_path=ann_path,
            vis_root=vis_root,
            modality_dropout_prob=modality_dropout_prob,
        )

        return datasets
