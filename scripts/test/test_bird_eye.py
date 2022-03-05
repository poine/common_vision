#!/usr/bin/env python3

import logging, glob, yaml
import numpy as np, cv2
import matplotlib.pyplot as plt

import common_vision.utils as cv_u
import common_vision.plot_utils as cv_pu
import common_vision.camera as cv_c
import common_vision.bird_eye as cv_be

import pdb
# TODO: draw triedra, write more about frames and coordinates

LOG = logging.getLogger('test_bird_eye')
logging.basicConfig(level=logging.INFO)


class BeParam:
    x0, y0, dx, dy = 0.29, 0, 4., 3. # bird eye area in local floor plane frame
    w = 640                    # bird eye image width (pixel coordinates)
    s = dy/w                   # scale
    h = int(dx/s)              # bird eye image height

class BeParamJulie:
    x0, y0, dx, dy = 2.7, 0, 15., 8.   # bird eye area in local floor plane frame
    w = 640                    # bird eye image width (pixel coordinates)
    s = dy/w                   # scale
    h = int(dx/s)              # bird eye image height

class BeParamSmocap:
    x0, y0, dx, dy = -0.5, -0.5, 1.5, 3. # bird eye area in local floor plane frame
#    x0, y0, dx, dy = 0., 0., 1., 1.      # bird eye area in local floor plane frame
    w = 1280                             # bird eye image width (pixel coordinates)
    s = dy/w                             # scale
    h = int(dx/s)                        # bird eye image height

    
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
        

def display_unwarped(be, cam, img_path):
    img =  cv2.imread(img_path, cv2.IMREAD_COLOR)
    #foo = np.array([[(0, 0), (200, 200), (100, 200)]])
    #cv2.polylines(img, foo, isClosed=True, color=(0, 0, 255), thickness=2)
    img2 = img.copy()
    
    cv2.polylines(img2, be.cam_img_mask, isClosed=True, color=(0, 0, 255), thickness=2)

    unwarped = be.undist_unwarp_img(img, cam)
    cv2.polylines(unwarped, be.unwarped_img_mask, isClosed=True, color=(0, 0, 255), thickness=2)
    
    cv2.imshow('camera', img2)
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
    img_path = '/home/poine/work/robot_data/christine/vedrines_track/frame_000000.png'
    display_unwarped(be, cam, img_path)
    

def test_julie():
    intr_cam_calib_path = '/home/poine/work/robot_data/julie/julie_short_range_intr.yaml'
    extr_cam_calib_path = '/home/poine/work/robot_data/julie/julie_short_range_extr.yaml'
    cam = cv_c.load_cam_from_files(intr_cam_calib_path, extr_cam_calib_path) 
    be = cv_be.BirdEye(cam, BeParamJulie())

    plot_bird_eye_2D(be)
    plt.show()
    img_path = '/home/poine/work/robot_data/julie/julie_extr_calib_1.png'
    display_unwarped(be, cam, img_path)

def test_smocap():
    intr_cam_calib_path = '/home/poine/work/smocap/smocap/params/enac_demo_z/ueye_enac_z_2.yaml'
    extr_cam_calib_path = '/home/poine/work/smocap/smocap/params/ricou/ueye_enac_z2_extr.yaml'
    cam = cv_c.load_cam_from_files(intr_cam_calib_path, extr_cam_calib_path) 
    be = cv_be.BirdEye(cam, BeParamSmocap())
    plot_bird_eye_2D(be)
    plt.show()
    img_path = '/home/poine/work/smocap/smocap/test/ricou/floor_cam_z2.png'
    display_unwarped(be, cam, img_path)

    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    LOG.info(" using opencv version: {}".format(cv2.__version__))
    #test_christine()
    #test_julie()
    test_smocap()
