import cv2
import numpy as np


class Evaluation:

    def convert2mask(self, mt, shape):
        # Converts coordinates of bounding-boxes into blank matrix with values set where bounding-boxes are.

        t = np.zeros([shape, shape])
        for m in mt:
            x, y, w, h = m
            cv2.rectangle(t, (x, y), (x + w, y + h), 1, -1)
        return t

    def prepare_for_detection(self, prediction, ground_truth):
        # For the detection task, convert Bounding-boxes to masked matrices (0 for background, 1 for the target). If you run segmentation, do not run this function

        if len(prediction) == 0:
            return [], []

        # Large enough size for base mask matrices:
        shape = 2 * max(np.max(prediction), np.max(ground_truth))

        p = self.convert2mask(prediction, shape)
        gt = self.convert2mask(ground_truth, shape)

        return p, gt

    def iou_compute(self, p, gt):
        # Computes Intersection Over Union (IOU)
        if len(p) == 0:
            return 0

        intersection = np.logical_and(p, gt)
        union = np.logical_or(p, gt)

        iou = np.sum(intersection) / np.sum(union)

        return iou

    # Add your own metrics here, such as mAP, class-weighted accuracy, ...
