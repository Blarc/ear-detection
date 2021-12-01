import cv2
import numpy as np
import os
import sys


class Detector:

	cascade_left = cv2.CascadeClassifier(
		os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_leftear.xml'))

	cascade_right = cv2.CascadeClassifier(
		os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_rightear.xml'))

	def detect(self, img_):
		det_list_left = self.cascade_left.detectMultiScale(img_, 1.05, 1)
		det_list_right = self.cascade_right.detectMultiScale(img_, 1.05, 1)

		if type(det_list_right) is not tuple and type(det_list_left) is not tuple:
			return np.concatenate((det_list_left, det_list_right), axis=0)
		if type(det_list_right) is not tuple:
			return det_list_right
		else:
			return det_list_left


if __name__ == '__main__':
	file_name = sys.argv[1]
	img = cv2.imread(file_name)
	detector = Detector()
	detected_loc = detector.detect(img)
	for x, y, w, h in detected_loc:
		cv2.rectangle(img, (x, y), (x + w, y + h), (128, 255, 0), 4)
	cv2.imwrite(file_name + '.detected.jpg', img)
