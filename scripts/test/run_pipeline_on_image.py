#!/usr/bin/env python
import sys, numpy as np, matplotlib.pyplot as plt, cv2
import pdb

#import common_vision.utils as cvu
import common_vision.camera as cvc
import two_d_guidance.trr.vision.lane_2 as trr_l2
import two_d_guidance.trr.vision.lane_3 as trr_l3
import common_vision.lane.lane_4 as lane_4

def test_on_img(pipe, cam, img):
    pipe.process_image(img, cam, None, None)
    out_img = pipe.draw_debug_bgr(cam)
    cv2.imshow('in', img); cv2.imshow('out', out_img)
    cv2.waitKey(0)
    
if __name__ == '__main__':
    robot_name = 'christine' 
    intr_cam_calib_path = '/home/poine/.ros/camera_info/{}_camera_road_front.yaml'.format(robot_name)
    extr_cam_calib_path = '/home/poine/work/oscar/oscar/oscar_description/cfg/{}_cam_road_front_extr.yaml'.format(robot_name)
    cam = cvc.load_cam_from_files(intr_cam_calib_path, extr_cam_calib_path)

    #pipe = trr_l2.Contour2Pipeline(cam, 'christine', ctr_img_min_area=50); # 500
    #pipe = trr_l3.Contour3Pipeline(cam, 'christine')
    pipe = lane_4.Pipeline(cam, 'christine')
    #pipe.thresholder.set_offset(10)
    #pipe.set_roi((0, 20), (cam.w, cam.h))
    #pipe.display_mode = trr_l2.Contour2Pipeline.show_contour
    pipe.display_mode = lane_4.Pipeline.show_summary
    
    img_path = '/home/poine/work/robot_data/christine/vedrines_track/frame_000000.png'
    if len(sys.argv) > 1: img_path = sys.argv[1]
    test_on_img(pipe, cam, cv2.imread(img_path, cv2.IMREAD_COLOR))
