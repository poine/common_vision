#!/usr/bin/env python
import os, sys, numpy as np
import roslib, rospy, rospkg, rostopic, cv_bridge, tf
import sensor_msgs.msg, cartographer_ros_msgs.msg, geometry_msgs.msg, visualization_msgs.msg
import cv2
import pdb

import common_vision.rospy_utils as cv_rpu
import common_vision.utils as cv_u


def msgPoint(x, y, z): p = geometry_msgs.msg.Point(); p.x=x; p.y=y; p.z=z; return p

class ArucoPipeline(cv_u.Pipeline):
    show_none, show_input = range(2)
    def __init__(self, cam, robot_name):
        cv_u.Pipeline.__init__(self)
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.aruco_parameters =  cv2.aruco.DetectorParameters_create()
        self.aruco_parameters.doCornerRefinement = False
        self.marker_size = 0.29
        self.img = None
        self.rvecs, self.tvecs, self.ids = [],[], []
        self.set_roi((0, 150), (cam.w, cam.h-400))
        
    def set_roi(self, tl, br):
        self.tl, self.br = tl, br
        self.roi_h, self.roi_w = self.br[1]-self.tl[1], self.br[0]-self.tl[0]
        self.roi = slice(self.tl[1], self.br[1]), slice(self.tl[0], self.br[0])
        
    def _process_image(self, img, cam, stamp):
        self.img = img
        self.corners, self.ids, self.rejectedImgPoints = cv2.aruco.detectMarkers(img[self.roi], self.aruco_dict, parameters=self.aruco_parameters)
        if self.ids is not None:
            for _cs in self.corners: _cs+= self.tl
            self.rvecs, self.tvecs = cv2.aruco.estimatePoseSingleMarkers(self.corners, self.marker_size, cam.K, cam.D)
        else:
            self.rvecs, self.tvecs, self.ids = [],[], []
            
    def draw_debug(self, cam, img_cam=None):
        return cv2.cvtColor(self.draw_debug_bgr(cam, img_cam), cv2.COLOR_BGR2RGB)

    def draw_debug_bgr(self, cam, img_cam=None):
        if self.img is None:
            return np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        else:
            debug_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(debug_img, tuple(self.tl), tuple(self.br), color=(0, 0, 255), thickness=3)
            if len(self.ids)>0: cv2.aruco.drawDetectedMarkers(debug_img, self.corners, self.ids)
            self.draw_timing(debug_img)
            return debug_img



class Node(cv_rpu.SimpleVisionPipeNode):

    def __init__(self):

        cv_rpu.SimpleVisionPipeNode.__init__(self, ArucoPipeline, self.pipe_cbk, 'mono8')
        self.pipeline.display_mode = self.pipeline.show_input
        
        # Debug Image publishing
        self.img_pub = cv_rpu.CompressedImgPublisher(self.cam, '/vision/aruco/image_debug')
        # Debug Markers publishing
        self.pose_pub = rospy.Publisher('/vision/aruco/pose', geometry_msgs.msg.PoseStamped, queue_size=1)
        self.marker_pub = rospy.Publisher('/vision/aruco/markers' , visualization_msgs.msg.MarkerArray, queue_size=1)
        # Landmarks publishing
        self.lm_pub = rospy.Publisher('/landmark', cartographer_ros_msgs.msg.LandmarkList, queue_size=1)

        self.start()

  

    def pipe_cbk(self):
        self.publish_landmarks(self.pipeline.rvecs, self.pipeline.tvecs, self.pipeline.ids)
        #print('node pipe')
        
    def periodic(self):
        if self.pipeline.display_mode != self.pipeline.show_none:
            self.img_pub.publish(self.pipeline, self.cam)
        if len(self.pipeline.ids) > 0:
            self.publish_pose(self.pipeline.rvecs, self.pipeline.tvecs, blf=False)
            self.publish_markers(self.pipeline.rvecs, self.pipeline.tvecs, self.pipeline.ids, blf=False)
        
    def publish_pose(self, rvecs, tvecs, blf=True):
        msg_pose = geometry_msgs.msg.PoseStamped()
        msg_pose.header.stamp = rospy.Time.now()
        T_lm_to_camo = cv_u.T_of_t_r(tvecs[0], rvecs[0])
        if blf:
            msg_pose.header.frame_id=self.ref_frame
            T_lm_to_blf = np.dot(self.cam.cam_to_world_T, T_lm_to_camo)
            cv_u._position_and_orientation_from_T(msg_pose.pose.position, msg_pose.pose.orientation, T_lm_to_blf)
        else:
            msg_pose.header.frame_id=self.cam.camo_frame
            cv_u._position_and_orientation_from_T(msg_pose.pose.position, msg_pose.pose.orientation, T_lm_to_camo)

        self.pose_pub.publish(msg_pose)

    def publish_markers(self, rvecs, tvecs, ids, rgba=(1.,1.,0.,1.), blf=False):
        msg = visualization_msgs.msg.MarkerArray()
        for _r, _t, _id in zip(rvecs, tvecs, ids):
            marker = visualization_msgs.msg.Marker()
            marker.type = marker.LINE_STRIP
            marker.action = marker.ADD
            marker.id = _id
            s = marker.scale; s.x, s.y, s.z = 0.01, 0, 0
            c = marker.color; c.r, c.g, c.b, c.a = rgba
            T_lm_to_camo = cv_u.T_of_t_r(_t, _r)
            if blf:
                marker.header.frame_id = self.ref_frame
                T_lm_to_blf = np.dot(self.cam.cam_to_world_T, T_lm_to_camo)
                cv_u._position_and_orientation_from_T(marker.pose.position, marker.pose.orientation, T_lm_to_blf)
            else:
                marker.header.frame_id = self.cam.camo_frame
                cv_u._position_and_orientation_from_T(marker.pose.position, marker.pose.orientation, T_lm_to_camo)
            d = self.pipeline.marker_size
            marker.points = [msgPoint(0, 0, 0), msgPoint(d, 0, 0), msgPoint(d, d, 0), msgPoint(0, d, 0), msgPoint(0, 0, 0)]
            msg.markers.append(marker)
        self.marker_pub.publish(msg)
        
    def publish_landmarks(self, rvecs, tvecs, ids):
        _msg = cartographer_ros_msgs.msg.LandmarkList() 
        _msg.header.stamp = rospy.Time.now()
        _msg.header.frame_id = self.ref_frame#"caroline/base_link_footprint"
        for _r, _v, _id in zip(rvecs, tvecs, ids):
            lme = cartographer_ros_msgs.msg.LandmarkEntry()
            #string id
            lme.id='{}'.format(_id)
            #geometry_msgs/Pose tracking_from_landmark_transform
            T_lm_to_camo = cv_u.T_of_t_r(_v, _r)
            T_lm_to_blf = np.dot(self.cam.cam_to_world_T, T_lm_to_camo)
            cv_u._position_and_orientation_from_T(lme.tracking_from_landmark_transform.position,
                                                  lme.tracking_from_landmark_transform.orientation, T_lm_to_blf)
            #float64 translation_weight
            #float64 rotation_weight
            lme.translation_weight = 0.1
            lme.rotation_weight = 0.1
            _msg.landmark.append(lme)
        self.lm_pub.publish(_msg)

        
def main(args):
  rospy.init_node('aruco_landmark_detector')
  Node().run(10)


if __name__ == '__main__':
    main(sys.argv)
