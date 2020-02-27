# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize

from .evaluator import DatasetEvaluator

from skimage.morphology import binary_dilation, disk, skeletonize


def db_eval_boundary(fg_boundary, gt_boundary, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.
    Arguments:
        fg_boundary (ndarray): binary boundary image.
        gt_boundary         (ndarray): binary annotated image.
    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th*np.linalg.norm(fg_boundary.shape))

    fg_dil = binary_dilation(fg_boundary,disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary,disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg     = np.sum(fg_boundary)
    n_gt     = np.sum(gt_boundary)

    # Compute precision and recall
    if n_fg == 0 and  n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0  and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match)/float(n_fg)
        recall    = np.sum(gt_match)/float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2*precision*recall/(precision+recall);
    return F


def db_eval_boundary(fg_boundary, gt_boundary, bound_th=0.008):
    out = [bu.db_eval_boundary(f, g, bound_th=bound_th) for f, g in zip(fg_boundary, gt_boundary)]
    return np.mean(out)


def _eval_lane(infos):
    gt_fn, pred_fn = infos
    gt = np.load(gt_fn)
    import pdb; pdb.set_trace()
    gt = gt.transpose(2,0,1)

    pred = np.load(pred_fn)
    out = []
    # binary F
    mask = skeletonize(pred[0] > 0)
    fs = [db_eval_boundary(mask, gt[0] > 0, bound_th=b) for b in [10, 5, 1]]



    # lane marking
    # convert to one-hot: 3 x C x H x W
    if len(pred) >= 3:
        pred_one_hot = [one_hot(p, c).transpose(2, 0, 1)[1:] for p, c in zip(pred[:3], [3, 3, 9])]
        gt_one_hot = [one_hot(g, c).transpose(2, 0, 1)[1:] for g, c in zip(gt[:3], [2, 3, 9])]

        # edge thinning
        lane_pred = [thin_edge(l) for l in pred_one_hot]
        # edge evaluation
        evals = []
        for bound_th in [10, 5, 1]:
            evals += [db_eval_boundary(l, t, bound_th=bound_th) for l, t in zip(lane_pred, gt_one_hot)]
        out.append(evals)


    return out


class LaneEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation
    """

    def __init__(self, dataset_name, distributed, num_classes, ignore_label=255, output_dir=None, task_name='lane'):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            num_classes (int): number of classes
            ignore_label (int): value in semantic segmentation ground truth. Predictions for the
            corresponding pixels should be ignored.
            output_dir (str): an output directory to dump results.
        """
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._num_classes = num_classes
        self._ignore_label = ignore_label
        self._N = num_classes + 1
        self.task_name = task_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["lane_file_name".format(self.task_name)]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None

    def reset(self):
        self._conf_matrix = np.zeros((self._N, self._N), dtype=np.int64)
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output_dir = output['lane_dir']
            output_cont = output['lane_cont']
            output_cat = output['lane_cat']
            pred = np.array(output, dtype=np.int)

            # edge evaluation
            evals = []
            for bound_th in [10, 5, 1]:
                evals += [db_eval_boundary(l, t, bound_th=bound_th) for l, t in zip(lane_pred, gt_one_hot)]
            out.append(evals)
            with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                gt = np.array(Image.open(f), dtype=np.int)
            # FIXME: temporary
            gt = gt[:, :, 0]
            gt[gt == self._ignore_label] = self._num_classese

            self._conf_matrix += np.bincount(
                self._N * pred.reshape(-1) + gt.reshape(-1), minlength=self._N ** 2
            ).reshape(self._N, self._N)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Binary boundary F measure
        * By-class mean boundary F measure
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "{}_predictions.json".format(self.task_name))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.zeros(self._num_classes, dtype=np.float)
        iou = np.zeros(self._num_classes, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc) / np.sum(acc_valid)
        miou = np.sum(iou) / np.sum(iou_valid)
        fiou = np.sum(iou * class_weights)
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "{}_evaluation.pth".format(self.task_name))
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({self.task_name: res})
        self._logger.info(results)
        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
            )
        return json_list
