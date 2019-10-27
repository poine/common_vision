#!/usr/bin/env python
import os, sys, numpy as np
import roslib, rospy, rospkg, rostopic, cv_bridge, tf
import sensor_msgs.msg, cartographer_ros_msgs.msg, geometry_msgs.msg, visualization_msgs.msg
import cv2

import common_vision.rospy_utils as cv_rpu
import common_vision.utils as cv_u


def msgPoint(x, y, z): p = geometry_msgs.msg.Point(); p.x=x; p.y=y; p.z=z; return p


class Node:

    def __init__(self):
        #intr_cam_calib_path = '/home/poine/.ros/camera_info/{}_camera_road_front.yaml'.format(robot_name)
        #extr_cam_calib_path = '/home/poine/work/oscar/oscar/oscar_description/cfg/{}_cam_road_front_extr.yaml'.format(robot_name)
        #cam = cvc.load_cam_from_files(intr_cam_calib_path, extr_cam_calib_path)

        robot_name = rospy.get_param('~robot_name', '')
        def prefix(robot_name, what): return what if robot_name == '' else '{}/{}'.format(robot_name, what)
        cam_name = rospy.get_param('~camera', prefix(robot_name, 'camera_horiz_front'))
        ref_frame = rospy.get_param('~ref_frame', prefix(robot_name, 'base_link_footprint'))

        self.cam = cv_rpu.retrieve_cam(cam_name, fetch_extrinsics=True, world=ref_frame)
        #self.cam.load_extrinsics('/home/poine/work/roverboard/roverboard_description/cfg/caroline_cam_road_front_extr.yaml')
        self.cam.set_undistortion_param(alpha=1.)

        
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.parameters =  cv2.aruco.DetectorParameters_create()
        #self.parameters.doCornerRefinement = False#True
        self.bridge = cv_bridge.CvBridge()
        self.image_pub = rospy.Publisher("/aruco_node/image_debug", sensor_msgs.msg.Image, queue_size=1)
        self.pose_pub = rospy.Publisher('/aruco_node/pose', geometry_msgs.msg.PoseStamped, queue_size=1)
        self.marker_pub = rospy.Publisher('/aruco_node/markers' , visualization_msgs.msg.MarkerArray, queue_size=1)
        self.lm_pub = rospy.Publisher('/landmark', cartographer_ros_msgs.msg.LandmarkList, queue_size=1)
        self.img_src_topic = '/caroline/camera_horiz_front/image_raw'
        self.img_sub = rospy.Subscriber(self.img_src_topic, sensor_msgs.msg.Image, self.img_callback,  queue_size = 1)

    def periodic(self):
        pass#self.publish_landmarks()
        
        
    def img_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
        except cv_bridge.CvBridgeError as e:
            print(e)

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.parameters)
        #print ids
        if ids is not None:
            marker_size = 1.
            rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, self.cam.K, self.cam.D)
            self.publish_pose(rvecs, tvecs)
        else:
            rvecs, tvecs, ids = [],[], []
        self.publish_markers(rvecs, tvecs, ids)
        self.publish_landmarks(rvecs, tvecs, ids)
        self.publish_image(cv_image, corners, ids)

    def publish_image(self, img, corners, ids):
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if len(ids)>0: cv2.aruco.drawDetectedMarkers(img_color, corners, ids)
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(img_color, "bgr8"))
        except cv_bridge.CvBridgeError as e:
            print(e)


            
    def publish_pose(self, rvecs, tvecs, blf=True):
        msg_pose = geometry_msgs.msg.PoseStamped()
        msg_pose.header.stamp = rospy.Time.now()
        if blf:
            msg_pose.header.frame_id='caroline/base_link_footprint'
            T_lm_to_camo = cv_u.T_of_t_r(tvecs[0], rvecs[0])
            T_lm_to_blf = np.dot(self.cam.cam_to_world_T, T_lm_to_camo)
            cv_u._position_and_orientation_from_T(msg_pose.pose.position, msg_pose.pose.orientation, T_lm_to_blf)
        else:
            msg_pose.header.frame_id='caroline/camera_horiz_front_optical_frame'
            T_lm_to_camo = cv_u.T_of_t_r(tvecs[0], rvecs[0])
            cv_u._position_and_orientation_from_T(msg_pose.pose.position, msg_pose.pose.orientation, T_lm_to_camo)
            

        self.pose_pub.publish(msg_pose)

    def publish_markers(self, rvecs, tvecs, ids, rgba=(1.,1.,0.,1.), blf=False):
        msg = visualization_msgs.msg.MarkerArray()
        for _r, _t, _id in zip(rvecs, tvecs, ids):
            marker = visualization_msgs.msg.Marker()
            marker.type = marker.LINE_STRIP
            marker.action = marker.ADD
            marker.id = _id
            #marker.text = 'foo'
            s = marker.scale; s.x, s.y, s.z = 0.01, 0, 0
            #marker.scale.x = 0.01
            c = marker.color; c.r, c.g, c.b, c.a = rgba
            T = cv_u.T_of_t_r(_t, _r)
            if blf:
                marker.header.frame_id = 'caroline/base_link_footprint'
                
                cv_u._position_and_orientation_from_T(marker.pose.position, marker.pose.orientation, T)
            else:
                marker.header.frame_id = 'caroline/camera_horiz_front_optical_frame'
                cv_u._position_and_orientation_from_T(marker.pose.position, marker.pose.orientation, T)
                
            marker.points = [msgPoint(0, 0, 0), msgPoint(1, 0, 0), msgPoint(1, 1, 0), msgPoint(0, 1, 0), msgPoint(0, 0, 0)]
            msg.markers.append(marker)
        self.marker_pub.publish(msg)
        
    def publish_landmarks(self, rvecs, tvecs, ids):
        _msg = cartographer_ros_msgs.msg.LandmarkList() 
        _msg.header.stamp = rospy.Time.now()
        _msg.header.frame_id = "caroline/base_link_footprint"
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
        
    def run(self, low_freq=20):
        rate = rospy.Rate(low_freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass

        
def main(args):
  rospy.init_node('aruco_landmark_publisher')
  Node().run()


if __name__ == '__main__':
    main(sys.argv)
