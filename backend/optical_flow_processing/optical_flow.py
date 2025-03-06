import cv2 as cv
import numpy as np


class OpticalFlowProcessing:
    def __init__(self):
        self.method = cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.prev_gray_img = None

    def __call__(self, img):
        img = img.astype(np.uint8)
        cur_gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if self.prev_gray_img is None:
            self.prev_gray_img = cur_gray_img.copy()

        flow = self.method.calc(self.prev_gray_img, cur_gray_img, None)

        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros_like(img)
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        flow_rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        flow_rgb = cv.GaussianBlur(flow_rgb, (5, 5), 0)

        self.prev_gray_img = cur_gray_img.copy()

        return flow_rgb

