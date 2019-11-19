import numpy as np
import rospy, tf2_ros, sensor_msgs.msg, cv_bridge, cv2

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

#
# Images
#
       
class ImgPublisher:
    def __init__(self, cam, img_topic = "/trr_vision/start_finish/image_debug"):
        rospy.loginfo(' publishing image on ({})'.format(img_topic))
        self.image_pub = rospy.Publisher(img_topic, sensor_msgs.msg.Image, queue_size=1)
        self.bridge = cv_bridge.CvBridge()
        
    def publish(self, producer, cam, encoding="rgb8"):
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(producer.draw_debug(cam), encoding))

class CompressedImgPublisher:
    def __init__(self, cam, img_topic):
        img_topic = img_topic + "/compressed"
        rospy.loginfo(' publishing image on ({})'.format(img_topic))
        self.image_pub = rospy.Publisher(img_topic, sensor_msgs.msg.CompressedImage, queue_size=1)
        
    def publish(self, model, data):
        img_rgb = model.draw_debug(data)
        self.publish2(img_rgb)
        
    def publish2(self, img_rgb):
        img_bgr =  cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        msg = sensor_msgs.msg.CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', img_bgr)[1]).tostring()
        self.image_pub.publish(msg)

class CameraListener:
    def __init__(self, cam_name, cbk=None, img_fmt="passthrough"):
        self.cam_name, self.img_cbk = cam_name, cbk
        self.bridge = cv_bridge.CvBridge()
        self.img, self.sub, self.img_fmt = None, None, img_fmt
        
    def start(self):
        self.cam_img_topic = '/{}/image_raw'.format(self.cam_name)
        rospy.loginfo(' -starting subscribtion to {}'.format(self.cam_img_topic))
        self.sub = rospy.Subscriber(self.cam_img_topic, sensor_msgs.msg.Image, self.img_callback, queue_size=1)

    def img_callback(self, msg):
        try:
            self.img = self.bridge.imgmsg_to_cv2(msg, self.img_fmt)
            if self.img_cbk is not None:
                self.img_cbk(self.img, (msg.header.stamp, msg.header.seq))
        except cv_bridge.CvBridgeError as e:
            print(e)

    def unregister(self):
         rospy.loginfo(' -stoping subscribtion to {}'.format(self.cam_img_topic))
         self.sub.unregister()
         self.sub = None

    def started(self): return self.sub is not None


class DebugImgPublisher:
    def __init__(self, cam_name, topic_sink):
        self.image_pub = CompressedImgPublisher(cam=None, img_topic=topic_sink)

        self.img, self.compressed_img = None, None
        self.img_src_topic = cam_name + '/image_raw/compressed'
        #self.img_sub = rospy.Subscriber(self.img_src_topic, sensor_msgs.msg.CompressedImage, self.img_cbk,  queue_size = 1)
        self.img_sub = None # in odred to keep network traffic down, we subscribe only when someone is listening to us
        self.compressed_img = None
        rospy.loginfo(' will subscribe to ({})'.format(self.img_src_topic))

    def img_cbk(self, msg):
        self.compressed_img = np.fromstring(msg.data, np.uint8)
        
    def publish(self, model, user_data):
        n_subscriber = self.image_pub.image_pub.get_num_connections()
        # don't bother drawing and publishing when no one is listening
        if n_subscriber <= 0:
            if self.img_sub is not None:
                self.img_sub.unregister()
                self.img_sub = None
                self.compressed_img = None
            return 
        else:
            if self.img_sub is None:
                self.img_sub = rospy.Subscriber(self.img_src_topic, sensor_msgs.msg.CompressedImage, self.img_cbk,  queue_size = 1)

        if self.compressed_img is not None:
            self.img_bgr = cv2.imdecode(self.compressed_img, cv2.IMREAD_COLOR)
            self._draw(self.img_bgr, model, user_data)
            self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
            #img_rgb = self.img[...,::-1] # rgb = bgr[...,::-1] OpenCV image to Matplotlib
            self.image_pub.publish2(self.img_rgb)

####
## Nodes

class PeriodicNode:

    def __init__(self, name):
        rospy.init_node(name)
    
    def run(self, freq):
        rate = rospy.Rate(freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass

    
class SimpleVisionPipeNode:
    def __init__(self, pipeline_class, pipe_cbk=None, img_fmt="passthrough"):
        robot_name = rospy.get_param('~robot_name', '')
        def prefix(robot_name, what): return what if robot_name == '' else '{}/{}'.format(robot_name, what)
        self.cam_name = rospy.get_param('~camera', prefix(robot_name, 'camera_road_front'))
        self.ref_frame = rospy.get_param('~ref_frame', prefix(robot_name, 'base_link_footprint'))

        self.cam = retrieve_cam(self.cam_name, fetch_extrinsics=True, world=self.ref_frame)
        self.cam.set_undistortion_param(alpha=1.)

        self.cam_lst = CameraListener(self.cam_name, self.on_image, img_fmt)
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


