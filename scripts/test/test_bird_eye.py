#!/usr/bin/env python

import logging, glob, yaml
import numpy as np, cv2
import matplotlib.pyplot as plt
import tf.transformations

import common_vision.utils as cv_u
import common_vision.plot_utils as cv_pu
import common_vision.camera as cv_c
import common_vision.bird_eye as cv_be

import pdb

LOG = logging.getLogger('test_bird_eye')
logging.basicConfig(level=logging.INFO)


class BeParam:
    x0, dx, dy = 0.29, 4., 3. # bird eye area in local floor plane frame
    w = 640                    # bird eye image width (pixel coordinates)
    s = dy/w                   # scale
    h = int(dx/s)              # bird eye image height


def plot_bird_eye_2D(be):
    ax = plt.gca()
    # plot a rough outline of the car
    car = np.array([(-0.1, 0.1), (0.2, 0.1), (0.2, -0.1), (-0.1, -0.1), (-0.1, 0.1)])
    plt.plot(car[:,0], car[:,1], label='car')
    # plot the outline of the camera viewing area
    plt.plot(be.cam_va_borders_fp_lfp[:,0], be.cam_va_borders_fp_lfp[:,1], label='camera viewing area')
    # plot the outline of the bird eye area
    plt.plot(be.corners_lfp[:,0], be.corners_lfp[:,1], label='bird eye area')
    # plot intersection of cam viewing area and bird eye area
    plt.plot(be.borders_isect_be_cam_lfp[:,0], be.borders_isect_be_cam_lfp[:,1], label='intersection') 
    ax.axis('equal')
    ax.xaxis.set_label_text('front')
    ax.yaxis.set_label_text('left')
    plt.legend()
        

def display_unwarped(be, cam):
    img_path = '/home/poine/work/robot_data/christine/vedrines_track/frame_000000.png'
    img =  cv2.imread(img_path, cv2.IMREAD_COLOR)
    unwarped = be.undist_unwarp_img(img, cam)
    #foo = np.array([[(0, 0), (200, 200), (100, 200)]])
    #cv2.polylines(img, foo, isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.polylines(img, be.cam_img_mask, isClosed=True, color=(0, 0, 255), thickness=2)
                    
    cv2.imshow('camera', img)
    cv2.imshow('unwarped', unwarped)
    cv2.waitKey(0)
    
def test_christine():
    # load camera intrinsics and extrinsics
    intr_cam_calib_path = '/home/poine/.ros/camera_info/christine_camera_road_front.yaml'
    extr_cam_calib_path = '/home/poine/work/oscar/oscar/oscar_description/cfg/christine_cam_road_front_extr.yaml'
    cam = cv_c.load_cam_from_files(intr_cam_calib_path, extr_cam_calib_path)

    be = cv_be.BirdEye(cam, BeParam())

    plot_bird_eye_2D(be)
    plt.show()
    display_unwarped(be, cam)
    


    


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    LOG.info(" using opencv version: {}".format(cv2.__version__))
    test_christine()
    
