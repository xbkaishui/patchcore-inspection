import contextlib
import gc
import logging
import os
from pathlib import Path
import sys

import click
import numpy as np
import torch

# todo for debug purposes only
src_path = os.path.join(os.path.dirname(__file__), "../src")
sys.path.append(src_path)

import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
from PIL import Image
import torch
from torchvision import transforms
from PyNomaly import loop
import time
from loguru import logger
import cv2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Predictor(object):

    def __init__(self, imagesize=224, resize=256, patch_core_path:str=None, good_image_path:str=None, 
                 good_image_load_size=6, **kwargs):
        # set device to gpu 0 
        self.device = patchcore.utils.set_torch_device([0])
        self.nn_method = patchcore.common.FaissNN(False, 8)
        self.patchcore_instance = patchcore.patchcore.PatchCore(self.device)
        self.patchcore_instance.load_from_path(
            load_path=patch_core_path, device=self.device, nn_method=self.nn_method
        )

        transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.loader = transforms.Compose(transform_img)
        good_images = []
        good_image_paths = [os.path.join(good_image_path, path)for path in os.listdir(good_image_path)]
        logger.info("good images: {}", good_image_paths)
        for img_path in good_image_paths[:good_image_load_size]:
            image = self.image_loader(img_path)
            image = image.squeeze()
            good_images.append(image)
        logger.info("load good images done")
        good_images = torch.stack(good_images)
        logger.info("run good images predict")
        scores, masks = self.patchcore_instance._predict(good_images)
        logger.info("predict good images done")
        self.good_scores = scores
        self.good_masks = masks

    def image_loader(self, img_path):
        """load image, returns cuda tensor"""
        image = Image.open(img_path)
        image = self.loader(image).float()
        image = image.unsqueeze(0)
        return image

    def predict(self, img_path):
        file_name = Path(img_path).stem
        model = self.patchcore_instance
        image = self.image_loader(img_path)
        original_image = image.squeeze().numpy().transpose(1,2,0) * 255
        cv2.imwrite(f"/tmp/{file_name}.png", original_image)
        start_time = time.time()
        preds, masks = model._predict(image)
        masks = np.array(masks)
        logger.info("preds mask shape: {}", masks.shape)
        min_mask = masks.reshape(-1).min()
        max_mask = masks.reshape(-1).max()
        segmentations = (masks - min_mask) / (max_mask - min_mask)
        segmentations = np.mean(segmentations, axis=0)
        segmentations = (segmentations > 0.5).astype(np.uint8) * 255
        logger.info("preds segmentations shape: {}", segmentations.shape)
        cv2.imwrite(f"/tmp/{file_name}_seg.png", segmentations)
        pred_scores = self.good_scores + preds
        m = loop.LocalOutlierProbability(np.array(pred_scores)).fit()
        prob_scores = m.local_outlier_probabilities
        confidence = prob_scores[-1]
        pred_class = "NG" if confidence > 0.5 else "Good"
        confidence = 1 - confidence
        pred_time = (time.time() - start_time) * 1000
        return {"class": pred_class, "good_probability": confidence, "time_taken" : int(pred_time)}
    
 

def predict_dir(predictor, image_dir):
    image_paths = [os.path.join(image_dir, path)for path in os.listdir(image_dir)]
    for img_path in image_paths:
        res = predictor.predict(img_path)
        logger.info("img_path {}, res: {}", img_path, res) 
    
if __name__ == "__main__":
    imagesize=484
    resize=484
    predictor = Predictor(imagesize=imagesize, resize=resize,
        patch_core_path="/opt/.pc/patchcore-inspection/snapshots/chip_f3_bottom", 
        good_image_path="/opt/.pc/mvtec/bottom/test/good")
    # logger.info("start predict good")
    # good_image_path = "/opt/.pc/mvtec/bottom/test/good"
    # predict_dir(predictor, good_image_path)
    # bad_image_path = "/opt/.pc/mvtec/bottom/test/NG"
    # predict_dir(predictor, bad_image_path)
    logger.info("start predict")
    res = predictor.predict("/opt/.pc/mvtec/bottom/test/NG/0029-B.jpg")
    res = predictor.predict("/opt/.pc/mvtec/bottom/test/NG/old-0100-B.jpg")
    logger.info("res: {}", res)