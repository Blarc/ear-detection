import os
import sys
import cv2
import torch

from homework2.utils.convert_annotations import denormalize


class Detector:
    weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), '', 'best.pt')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

    # By calling with image_name, the model resizes the picture as appropriate
    def detect(self, image_name):
        detections = []
        results = self.model(image_name, size=416)
        for tensor1, tensor2 in zip(results.xywhn, results.xywh):
            for result1, result2 in zip(tensor1, tensor2):
                x_norm, y_norm, w_norm, h_norm, _, _ = result1.numpy()
                x_, y_, w_, h_, _, _ = result2.numpy()
                detections.extend([denormalize(x_norm, y_norm, w_norm, h_norm)])

                # denormalize(results.xywh) != results.xywh !!
                # print(denormalize(x_norm, y_norm, w_norm, h_norm))
                # print(x_, y_, w_, h_)

        return detections


if __name__ == '__main__':
    file_name = sys.argv[1]
    img = cv2.imread(file_name)
    detector = Detector()
    detected_loc = detector.detect(file_name)
    for x, y, w, h in detected_loc:
        cv2.rectangle(img, (x, y), (x + w, y + h), (128, 255, 0), 4)
    cv2.imwrite(file_name + '.detected.jpg', img)
