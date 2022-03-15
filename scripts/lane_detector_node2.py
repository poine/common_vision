#!/usr/bin/env python3

import sys, numpy as np, rospy, dynamic_reconfigure.server
import cv2

# ./scripts/lane_detector_node2.py  _camera:=camera1 _robot_name:=trilopi _ref_frame:=base_link_footprint


import common_vision.cfg.lane_detectorConfig

import common_vision.rospy_utils as cv_rpu
import common_vision.utils as cv_u

import common_vision.lane.lane_1 as cv_l
import common_vision.lane.lane_2 as cv_l2

class Node(cv_rpu.SimpleVisionPipeNode):

    def __init__(self, compress_debug=True):
        pipe_args = {'cache_filename':'/tmp/be_calib.npz', 'force_recompute':False}
        #pipe_class = cv_l.Contour1Pipeline
        pipe_class = cv_l2.Lane2Pipeline
        cv_rpu.SimpleVisionPipeNode.__init__(self, pipe_class, self.pipe_cbk, pipe_args,
                                             img_fmt="passthrough", fetch_extrinsics=True)
        self.lane_model_pub = cv_rpu.LaneModelPublisher('/vision/lane/model', who='vision_lane_node')
        #self.pipeline.display_mode = cv_l.Contour1Pipeline.show_thresh
        args = (self.cam, '/vision/lane/image')
        self.img_pub = cv_rpu.CompressedImgPublisher(*args) if compress_debug else cv_rpu.ImgPublisher(*args)
        self.cfg_srv = dynamic_reconfigure.server.Server(common_vision.cfg.lane_detectorConfig, self.cfg_callback)
       
        self.start()

    def cfg_callback(self, config, level):
        rospy.loginfo("  Reconfigure Request:")
        #try:
        #    self.pipeline.contour_finder.min_area = config['mask_min_area']
        #    self.pipeline.thresholder.set_offset(config['bridge_offset'])
        #except AttributeError: pass
        self.pipeline.display_mode = config['display_mode']
        return config

    def pipe_cbk(self):
        if self.pipeline.lane_model.is_valid():
            self.lane_model_pub.publish(self.pipeline.lane_model)

    def periodic(self):
        #print('proc: {:.1f}ms'.format(self.pipeline.lp_proc*1e3))
        if self.pipeline.display_mode != self.pipeline.show_none:
            self.img_pub.publish(self.pipeline, self.cam)
            
    def draw_debug(self, cam, img_cam=None):
        return self.pipeline.img

def main(args):
    name = 'vision_lane_detector_node'
    rospy.init_node(name)
    rospy.loginfo('{} starting'.format(name))
    rospy.loginfo('  using opencv version {}'.format(cv2.__version__))
    Node().run(low_freq=4)


if __name__ == '__main__':
    main(sys.argv)
