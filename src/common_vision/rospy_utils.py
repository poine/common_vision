import numpy as np
import rospy, tf2_ros, sensor_msgs.msg, cv_bridge

import common_vision.camera as cv_cam
import common_vision.utils as cv_u

def retrieve_cam(cam_name, _id=0, fetch_extrinsics=True, world='world'):
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.loginfo(' retrieving camera: "{}" configuration'.format(cam_name))
    cam = cv_cam.Camera(_id, cam_name)
    # Retrieve camera instrinsic 
    cam_info_topic = '/{}/camera_info'.format(cam_name)
    rospy.loginfo(' -retrieving intrinsics on topic: {}'.format(cam_info_topic))
    cam_info_msg = rospy.wait_for_message(cam_info_topic, sensor_msgs.msg.CameraInfo)
    cam.set_calibration(np.array(cam_info_msg.K).reshape(3,3), np.array(cam_info_msg.D), cam_info_msg.width, cam_info_msg.height)
    rospy.loginfo('   retrieved intrinsics ({})'.format(cam_info_topic))
    # Retrieve camera extrinsic
    if fetch_extrinsics:
        cam_frame = '{}_optical_frame'.format(cam.name)
        rospy.loginfo(' -retrieving extrinsics ( {} to {} )'.format(world, cam_frame))
        while not cam.is_localized():
            try:
                world_to_camo_transf = tf_buffer.lookup_transform(target_frame=cam_frame, source_frame=world, time=rospy.Time(0))
                world_to_camo_t, world_to_camo_q = cv_u.t_q_of_transf_msg(world_to_camo_transf.transform)
                cam.set_location(world_to_camo_t, world_to_camo_q)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.loginfo_throttle(1., " waiting to get camera location {}".format(e))
        rospy.loginfo('   retrieved extrinsics (w2co_t {} w2co_q {})'.format(np.array2string(np.asarray(world_to_camo_t), precision=3),
                                                                             np.array2string(np.asarray(world_to_camo_q), precision=4)))
    rospy.loginfo(' -retrieving camera encoding')
    cam_img_topic = '/{}/image_raw'.format(cam_name)
    cam_img_msg = rospy.wait_for_message(cam_img_topic, sensor_msgs.msg.Image)
    cam.set_encoding(cam_img_msg.encoding)
    rospy.loginfo('   retrieved encoding ({})'.format(cam_img_msg.encoding))
    return cam


class CameraListener:
    def __init__(self, cam_name, cbk=None):
        self.cam_name, self.img_cbk = cam_name, cbk
        self.bridge = cv_bridge.CvBridge()
        self.img = None
        self.sub = None
        
    def start(self):
        self.cam_img_topic = '/{}/image_raw'.format(self.cam_name)
        rospy.loginfo(' -starting subscribtion to {}'.format(self.cam_img_topic))
        self.sub = rospy.Subscriber(self.cam_img_topic, sensor_msgs.msg.Image, self.img_callback, queue_size=1)

    def img_callback(self, msg):
        try:
            self.img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            if self.img_cbk is not None:
                self.img_cbk(self.img, (msg.header.stamp, msg.header.seq))
        except cv_bridge.CvBridgeError as e:
            print(e)

    def unregister(self):
         rospy.loginfo(' -stoping subscribtion to {}'.format(self.cam_img_topic))
         self.sub.unregister()
         self.sub = None

    def started(self): return self.sub is not None
            
class SimpleVisionPipeNode:
    def __init__(self, pipeline_class, pipe_cbk=None):
        robot_name = rospy.get_param('~robot_name', '')
        def prefix(robot_name, what): return what if robot_name == '' else '{}/{}'.format(robot_name, what)
        self.cam_name = rospy.get_param('~camera', prefix(robot_name, 'camera_road_front'))
        self.ref_frame = rospy.get_param('~ref_frame', prefix(robot_name, 'base_link_footprint'))

        self.cam = retrieve_cam(self.cam_name, fetch_extrinsics=True, world=self.ref_frame)
        self.cam.set_undistortion_param(alpha=1.)

        self.cam_lst = CameraListener(self.cam_name, self.on_image)
        self.pipeline = pipeline_class(self.cam, robot_name)


        
    def start(self):
        self.cam_lst.start()

    def started(self): return self.cam_lst.started()
    
    def stop(self):
        self.cam_lst.unregister()
        
    # we get a bgr8 image as input
    def on_image(self, img_bgr, (stamp, seq)):
        self.pipeline.process_image(img_bgr, self.cam, stamp, seq)
        if self.pipe_cbk is not None: self.pipe_cbk()
        
    def run(self, low_freq=10):
        rate = rospy.Rate(low_freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass


