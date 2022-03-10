#!/usr/bin/env python3

# ./scripts/bird_eye_node.py  _camera:=camera1 _robot_name:=trilopi _ref_frame:=base_link_footprint

import sys, numpy as np, rospy, dynamic_reconfigure.server
import cv2

import common_vision.cfg.bird_eye_nodeConfig

import common_vision.rospy_utils as cv_rpu
import common_vision.utils as cv_u
import common_vision.bird_eye as cv_be

class BeParamTrilopi:
    x0, y0, dx, dy = 0.11, 0., 0.25, 0.2 # bird eye area in local floor plane frame
    w = 640                     # bird eye image width (pixel coordinates)
    s = dy/w                    # scale
    h = int(dx/s)               # bird eye image height

    
class BirdEyePipeline(cv_u.Pipeline):
    show_none=0
    def __init__(self, cam, robot_name):
        cv_u.Pipeline.__init__(self)
        #print(f'ros extrinsics\n{cam.world_to_cam_T}')
        print(f'ros world_to_cam\n{cam.world_to_cam_t} {cam.world_to_cam_q}')
        #be_param = cv_be.BirdEye(cam, BeParam())#trr_vu.NamedBirdEyeParam(robot_name)
        intr_cam_calib_path = '/home/ubuntu/work/robot_data/trilopi/camera1_intrinsics2.yaml'
        #extr_cam_calib_path = '/home/ubuntu/work/robot_data/trilopi/camera1_extrinsics.yaml'
        extr_cam_calib_path = '/tmp/extcalib.yaml'
        ##cam.load_all({'intrinsics': intr_cam_calib_path, 'extrinsics)
        #cam.load_intrinsics(intr_cam_calib_path)
        cam.load_extrinsics(extr_cam_calib_path)
        print(f'loaded world_to_cam\n{cam.world_to_cam_t} {cam.world_to_cam_q}')
        self.bird_eye = cv_be.BirdEye(cam, BeParamTrilopi())
        self.display_mode = 2
        self.img = None


        
    def _process_image(self, img, cam, stamp):
        self.img = img
        self.img_unwarped = self.bird_eye.undist_unwarp_img(self.img, cam)

    def draw_debug_bgr(self, cam, img_cam=None):
        if self.img is None:
            return np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        else:
            if self.display_mode == 1:
                debug_img = self.img.copy()
                #debug_img = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
            else:
                debug_img = self.img_unwarped
            #cv2.rectangle(debug_img, tuple(self.tl), tuple(self.br), color=(0, 0, 255), thickness=3)
            self.draw_timing(debug_img)
            return debug_img

        

class Node(cv_rpu.SimpleVisionPipeNode):

    def __init__(self):
       cv_rpu.SimpleVisionPipeNode.__init__(self, BirdEyePipeline, self.pipe_cbk, img_fmt="passthrough", fetch_extrinsics=True)
       self.img_pub = cv_rpu.ImgPublisher(self.cam, 'vision/bird_eye')
       self.cfg_srv = dynamic_reconfigure.server.Server(common_vision.cfg.bird_eye_nodeConfig, self.cfg_callback)
       self.start()

    def cfg_callback(self, config, level):
        rospy.loginfo("  Reconfigure Request:")
        self.pipeline.display_mode = config['display_mode']
        return config

    def pipe_cbk(self):
        #self.img_pub.publish(self, self.cam, "bgr8")
        pass

    def periodic(self):
        #print('proc: {:.1f}ms'.format(self.pipeline.lp_proc*1e3))
        if self.pipeline.display_mode != self.pipeline.show_none:
            self.img_pub.publish(self.pipeline, self.cam)
            
    def draw_debug(self, cam, img_cam=None):
        return self.pipeline.img

def main(args):
    name = 'vision_bird_eye_node'
    rospy.init_node(name)
    rospy.loginfo('{} starting'.format(name))
    rospy.loginfo('  using opencv version {}'.format(cv2.__version__))
    Node().run(low_freq=2)


if __name__ == '__main__':
    main(sys.argv)
