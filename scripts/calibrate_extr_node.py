#!/usr/bin/env python

import sys, numpy as np, rospy
import cv2


import common_vision.rospy_utils as cv_rpu
import common_vision.utils as cv_u

import two_d_guidance.trr.vision.utils as trr_vu


import pdb

class ImgPublisher(cv_rpu.DebugImgPublisher):
    def __init__(self, img_topic, cam_name):
        cv_rpu.DebugImgPublisher.__init__(self, cam_name, img_topic)

    def _draw(self, img_bgr, model, data):
        y0=20; font_color=(128,0,255)
        f, h1, h2, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1.25, 0.9, font_color, 2
        cv2.putText(img_bgr, 'Calibration:', (y0, 40), f, h1, c, w)

        contour_lfp = np.array([[0.29, -0.2, 0],
                                [1.5, -1.4 , 0],
                                [1.5*np.linalg.norm([1.5, 1.4]), 0, 0],
                                [1.5,  1.4 , 0],
                                [0.15,  0.23, 0]])
        contour_lfp = np.array([[0.208, -0.15, 0],
                                [1.5, -1.4 , 0],
                                [1.5*np.linalg.norm([1.5, 1.4]), 0, 0],
                                [1.5,  1.4 , 0],
                                [0.20,  0.165, 0]])
        contour_img = model.cam.project(contour_lfp).astype(np.int64)
        cv2.polylines(img_bgr, [contour_img] , isClosed=True, color=(0, 255, 255), thickness=2)
        # Draw keypoint on image, write their names and coordinates
        pts_name, pts_img, pts_world = model.pts_name, model.pts_img, model.pts_world
        for i, p in enumerate(pts_img.astype(int)):
            cv2.circle(img_bgr, tuple(p), 1, (0,255,0), -1)
            cv2.putText(img_bgr, '{}'.format(pts_name[i][:1]), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.putText(img_bgr, '{}'.format(pts_world[i][:2]), tuple(p+[0, 25]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        for i, p in enumerate(model.rep_pts.astype(int)):
            cv2.circle(img_bgr, tuple(p), 1, (0,0,255), -1)
        #print model, data
        
class NonePipeline(trr_vu.Pipeline):
    def __init__(self, cam, robot_name):
        self.cam = cam
        trr_vu.Pipeline.__init__(self)
        #be_param = trr_vu.NamedBirdEyeParam(robot_name)
        #self.bird_eye = trr_vu.BirdEyeTransformer(cam, be_param)
        
    def _process_image(self, img, cam, stamp):
        self.img = img
        #self.img = self.bird_eye.process(img)
        

class Node(cv_rpu.SimpleVisionPipeNode):

    def __init__(self):
        # will load a camera (~camera and ~ref_frame)
        cv_rpu.SimpleVisionPipeNode.__init__(self, NonePipeline, self.pipe_cbk)
        self.img_pub = cv_rpu.ImgPublisher(self.cam, '/vision/calibrate_extr')
        self.img_pub2 = ImgPublisher("/vision/calibrate_2", self.cam_name)
        # load keypoints
        points_path = '/home/poine/work/roverboard/roverboard_caroline/config/ext_calib_pts_floor_tiles_ricou_01.yaml'
        self.pts_name, self.pts_img, self.pts_world = cv_u.read_point(points_path)
       
        self.rep_pts = self.cam.project(self.pts_world).squeeze()
        #pdb.set_trace()
        self.start()

    def pipe_cbk(self):
        self.img_pub.publish(self, self.cam, "bgr8")

    def periodic(self):
        print('proc: {:.1f}ms'.format(self.pipeline.lp_proc*1e3))
        self.img_pub2.publish(self, None)
        
    def draw_debug(self, cam, img_cam=None):
        return self.pipeline.img

        
def main(args):
    name = 'vision_calibrate_extrinsic_node'
    rospy.init_node(name)
    rospy.loginfo('{} starting'.format(name))
    rospy.loginfo('  using opencv version {}'.format(cv2.__version__))
    Node().run(low_freq=2)


if __name__ == '__main__':
    main(sys.argv)
