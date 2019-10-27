#!/usr/bin/env python
import numpy as np, cv2, matplotlib.pyplot as plt
import logging
LOG = logging.getLogger('calibrate_intrisinc')
import common_vision.utils as cvu
import common_vision.plot_utils as cvpu
import common_vision.camera as cvc


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    LOG.info(" using opencv version: {}".format(cv2.__version__))

    intr_cam_calib_path = '/home/poine/.ros/camera_info/caroline_camera_one.yaml'
    #extr_cam_calib_path = '/home/poine/work/oscar/oscar/oscar_description/cfg/{}_cam_road_front_extr.yaml'.format(robot_name)
    cam = cvc.load_cam_from_files(intr_cam_calib_path, extr_path=None)

    img = cv2.imread('/home/poine/work/robot_data/caroline/floor_tiles_ricou_01.png')
    #img2 = cam.undistort_img(img)
    img2 = cam.undistort_img_map(img)
    #img2 = cv2.undistort(img, camera_matrix, dist_coeffs, None, undist_camera_matrix)
    cv2.imshow('distorted image', img)
    cv2.imshow('undistorted image', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
