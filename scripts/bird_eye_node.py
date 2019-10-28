#!/usr/bin/env python

import sys, numpy as np, rospy
import cv2


import common_vision.rospy_utils as cv_rpu
import two_d_guidance.trr.vision.utils as trr_vu

class BirdEyePipeline(trr_vu.Pipeline):
    def __init__(self, cam, robot_name):
        trr_vu.Pipeline.__init__(self)
        be_param = trr_vu.NamedBirdEyeParam(robot_name)
        self.bird_eye = trr_vu.BirdEyeTransformer(cam, be_param)
        
    def _process_image(self, img, cam, stamp):
        self.img = self.bird_eye.process(img)


class Node(cv_rpu.SimpleVisionPipeNode):

    def __init__(self):
       cv_rpu.SimpleVisionPipeNode.__init__(self, BirdEyePipeline, self.pipe_cbk)
       self.img_pub = cv_rpu.ImgPublisher(self.cam, 'trr/vision/bird_eye')
       self.start()

    def pipe_cbk(self):
        self.img_pub.publish(self, self.cam, "bgr8")

    def periodic(self):
        print('proc: {:.1f}ms'.format(self.pipeline.lp_proc*1e3))

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
