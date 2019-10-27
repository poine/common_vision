#!/usr/bin/env python
import os, sys, numpy as np
import roslib, rospy, rospkg, rostopic, cv_bridge, tf
import cv2

import common_vision.rospy_utils as cv_rpu
import common_vision.utils as cv_u
#import smocap.rospy_utils
import pdb

def test_mask(cam, img):


    contour_lfp = np.array([[0.5, -0.38, 0],
                            [1.5, -1.4 , 0],
                            [np.linalg.norm([1.5, 1.4]), 0, 0],
                            [1.5,  1.4 , 0],
                            [0.5,  0.38, 0]])
    mask = cv_u.Mask(cam, contour_lfp)
    # contour_img = cam.project(contour_lfp).astype(np.int64).squeeze()
    # mask = np.zeros(img.shape[:2], np.uint8)
    # cv2.fillPoly(mask, [contour_img], color=255)
    out_img = cv2.bitwise_and(img, img, mask=mask.mask)

    
    #y0, x1, y1, x2, y3 = 350, 150, cam.h-20, 600, 350
    #contour_img = np.array( [ [0, cam.h], [0, y0], [x1, y1], [x2,y1], [cam.w, y3],  [cam.w, cam.h] ] )
    #pdb.set_trace()
    cv2.polylines(img, [mask.contour_img] , isClosed=True, color=(0, 255, 255), thickness=2)
    #cv2.fillPoly(img, [contour_img], color=(0,0,0))
    #out_img = pipe.draw_debug_bgr(cam)
    cv2.imshow('in', img); cv2.imshow('out', out_img)
    cv2.waitKey(0)


def draw_3D(cam):
    pass

def retrieve_cam(cam_name = 'caroline/camera_horiz_front', ref_frame = 'caroline/base_link_footprint'):
    rospy.init_node('test_cam')
    cam = cv_rpu.retrieve_cam(cam_name, fetch_extrinsics=True, world=ref_frame)
    #cam_sys = smocap.rospy_utils.CamSysRetriever().fetch([cam_name], fetch_extrinsics=True, world=ref_frame)
    return cam

def main():
    cam = retrieve_cam()
    img_path = '/home/poine/work/robot_data/caroline/gazebo/horiz_image_01.png'
    if len(sys.argv) > 1: img_path = sys.argv[1]
    test_mask(cam, cv2.imread(img_path, cv2.IMREAD_COLOR))

    
if __name__ == '__main__':
    main()
