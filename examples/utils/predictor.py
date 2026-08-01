import torch

from detectron2.data import MetadataCatalog

from torch import nn
import detectron2.data.transforms as T
from detectron2.config import configurable
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.backbone.backbone import Backbone

from detectron2.structures import ImageList
from detectron2.modeling import (build_backbone, META_ARCH_REGISTRY,
                                 build_proposal_generator, build_roi_heads,
                                 detector_postprocess)
from typing import Optional, Tuple
from collections import OrderedDict
import torch.nn.functional as F
import math, torch
def _crop_one_level_proportional(feat: torch.Tensor, H_in:int, W_in:int, H_pad:int, W_pad:int):

    if H_in == H_pad and W_in == W_pad:
        return feat

    Hf, Wf = feat.shape[-2], feat.shape[-1]
    b = H_pad - H_in
    r = W_pad - W_in

    cols_crop = int(round(Wf * (r / max(1, W_pad))))

    if rows_crop == 0 and cols_crop == 0:
        return feat

    rows_crop = min(rows_crop, Hf)
    cols_crop = min(cols_crop, Wf)
    return feat[..., :Hf - rows_crop, :Wf - cols_crop]

def crop_pyramid_to_valid_proportional(feats, H_in:int, W_in:int, H_pad:int, W_pad:int):
    if isinstance(feats, torch.Tensor):
        return _crop_one_level_proportional(feats, H_in, W_in, H_pad, W_pad)
    elif isinstance(feats, (list, tuple)):
        return type(feats)(_crop_one_level_proportional(f, H_in, W_in, H_pad, W_pad) for f in feats)
    elif isinstance(feats, dict):
        return {k: _crop_one_level_proportional(v, H_in, W_in, H_pad, W_pad) for k, v in feats.items()}
    else:
        raise TypeError(f"Unsupported feats type: {type(feats)}")
class ResNetFPNBody(nn.Module):

    def __init__(self, fpn_model: nn.Module):

        super().__init__()

        self.bottom_up = fpn_model.bottom_up

        self.fpn_output2 = fpn_model.fpn_output2
        self.fpn_lateral3 = fpn_model.fpn_lateral3
        self.fpn_output3 = fpn_model.fpn_output3
        self.fpn_lateral4 = fpn_model.fpn_lateral4
        self.fpn_output4 = fpn_model.fpn_output4
        self.fpn_lateral5 = fpn_model.fpn_lateral5
        self.fpn_output5 = fpn_model.fpn_output5

        self.top_block = fpn_model.top_block

    def forward(self, x_stem: torch.Tensor) -> dict:

        features = {}
        x = self.bottom_up.res2(x_stem)
        features['res2'] = x
        x = self.bottom_up.res3(x)
        features['res3'] = x
        x = self.bottom_up.res4(x)
        features['res4'] = x
        x = self.bottom_up.res5(x)
        features['res5'] = x

        p5_out = self.fpn_output5(p5_in)

        p4_in_from_res4 = self.fpn_lateral4(features['res4'])
        p4_in_from_p5 = F.interpolate(p5_in, size=p4_in_from_res4.shape[-2:], mode="nearest")
        p4_in = p4_in_from_res4 + p4_in_from_p5
        p4_out = self.fpn_output4(p4_in)

        p3_in_from_res3 = self.fpn_lateral3(features['res3'])
        p3_in_from_p4 = F.interpolate(p4_in, size=p3_in_from_res3.shape[-2:], mode="nearest")
        p3_in = p3_in_from_res3 + p3_in_from_p4
        p3_out = self.fpn_output3(p3_in)

        p2_in_from_res2 = self.fpn_lateral2(features['res2'])
        p2_in_from_p3 = F.interpolate(p3_in, size=p2_in_from_res2.shape[-2:], mode="nearest")
        p2_in = p2_in_from_res2 + p2_in_from_p3
        p2_out = self.fpn_output2(p2_in)

        p6_out = self.top_block(p5_out)[0]

        results = {
            "p2": p2_out,
            "p3": p3_out,
            "p4": p4_out,
            "p5": p5_out,
            "p6": p6_out,
        }
        return results
import  copy
@META_ARCH_REGISTRY.register()
class GeneralizedRCNN_with_Rate(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):

        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, trand_y_tilde):

        if not self.training:
            return self.inference(batched_inputs, trand_y_tilde=trand_y_tilde)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features, distortion, rate = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses, distortion, rate
    def forward_1(self, batched_inputs, trand_y_tilde,pad_shape):

        if not self.training:
            return self.inference_1(batched_inputs, trand_y_tilde=trand_y_tilde,pad_shape=pad_shape)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features, distortion, rate = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses, distortion, rate
    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True, trand_y_tilde=None):

        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(trand_y_tilde)
        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return self._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results
    def inference_1(self, batched_inputs, detected_instances=None, do_postprocess=True, trand_y_tilde=None,pad_shape =None):

        assert not self.training
        body_net = ResNetFPNBody(copy.deepcopy(self.backbone))
        images = self.preprocess_image(batched_inputs)
        features = body_net(trand_y_tilde)

        H_in, W_in, H_pad, W_pad = pad_shape
        features = crop_pyramid_to_valid_proportional(features, H_in, W_in, H_pad, W_pad)
        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return self._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs):

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

class ModPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, trand_y_tilde):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image[0], "height": height, "width": width}
            predictions = self.model([inputs], trand_y_tilde)[0]
            return predictions

class FeaturePredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, trand_y_tilde,pad_shape):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image[0], "height": height, "width": width}
            predictions = self.model.forward_1([inputs], trand_y_tilde,pad_shape)[0]
            return predictions

