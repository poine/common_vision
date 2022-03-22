import numpy as np
import rospy, tf2_ros, sensor_msgs.msg
import cv_bridge, cv2
import sensor_msgs.msg, geometry_msgs.msg, visualization_msgs.msg #, tf

import common_vision.msg
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
# Generic
#
   
class NoRXMsgException(Exception): pass
class RXMsgTimeoutException(Exception): pass

class SimplePublisher(rospy.Publisher):
    def __init__(self, topic, msg_class, what, qs=1):
        rospy.loginfo(' {} publishing on {}'.format(what, topic))
        rospy.Publisher.__init__(self, topic, msg_class, queue_size=qs)
        self.msg_class = msg_class
        
class SimpleSubscriber:
    def __init__(self, topic, msg_class, what, timeout=0.5, user_cbk=None):
        self.sub = rospy.Subscriber(topic, msg_class, self.msg_callback, queue_size=1)
        rospy.loginfo(' {} subscribed to {}'.format(what, topic))
        self.timeout, self.user_cbk = timeout, user_cbk
        self.msg = None
        
    def msg_callback(self, msg):
        self.msg = msg
        self.last_msg_time = rospy.get_rostime()
        if self.user_cbk is not None: self.user_cbk(msg)

    def get(self):
        if self.msg is None:
            raise NoRXMsgException
        if (rospy.get_rostime()-self.last_msg_time).to_sec() > self.timeout:
            raise RXMsgTimeoutException
        return self.msg




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

#
# A publisher that only publishes when someone is listening
# It also handles fetching images from a camera topic when publishing
#
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
            self._draw(self.img_bgr, model, user_data) # needs to be implemented by superclasses
            self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
            #img_rgb = self.img[...,::-1] # rgb = bgr[...,::-1] OpenCV image to Matplotlib
            self.image_pub.publish2(self.img_rgb)


#
# A publisher that only publishes when someone is listening
#
class SavyPublisher(SimplePublisher):
    def __init__(self, topic, msg_class, what, qs=1):
        SimplePublisher.__init__(self, topic, msg_class, what, qs)

    def publish1(self, model, args):
        n_subscriber = self.get_num_connections()
        if n_subscriber <= 0:
            if self._is_connected(): self._disconnect()
        else:
            if not self._is_connected(): self._connect()
            self._publish(model, args)

        

### Transforms, stolen from pat3 to avoid dependency


class TransformPublisher:
    def __init__(self):
        self.tfBcaster = tf2_ros.TransformBroadcaster()
        # FIXME, use that for ENU to NED
        self.static_tfBcaster = tf2_ros.StaticTransformBroadcaster()
        self.tfBuffer  = tf2_ros.Buffer()
        self.tfLstener = tf2_ros.TransformListener(self.tfBuffer)

    def publish(self, t, T_w2b):
        self.send_w_enu_to_ned_transform(t)
        if T_w2b is not None: self.send_w_ned_to_b_transform(t, T_w2b) 
    
    def send_w_enu_to_ned_transform(self, t):
        R_enu2ned = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        T_enu2ned = np.eye(4); T_enu2ned[:3,:3] = R_enu2ned
        self.send_transform("w_enu", "w_ned", t, T_enu2ned, static=True)

    def send_w_ned_to_b_transform(self, t, T_w2b):
        self.send_transform("w_ned", "b_frd", t, T_w2b)

    def send_w_enu_to_b_transform(self, t, T_w2b):
        self.send_transform("w_enu", "b_frd", t, T_w2b)

    def send_b_to_a_transform(self, t, T_b2a):
        self.send_transform("b_frd", "a_ab", t, T_b2a)
        
    def send_transform(self, f1, f2, t, T_f1tof2, static=False):
        tf_msg = geometry_msgs.msg.TransformStamped()
        tf_msg.header.frame_id = f1
        tf_msg.child_frame_id = f2
        tf_msg.header.stamp = t
        _r = tf_msg.transform.rotation
        
        _r.x, _r.y, _r.z, _r.w = cv_u.quaternion_from_matrix(T_f1tof2)#tf.transformations.quaternion_from_matrix(T_f1tof2)
        _t = tf_msg.transform.translation
        _t.x, _t.y, _t.z = T_f1tof2[:3,3]
        if static:
            self.static_tfBcaster.sendTransform(tf_msg)
        else:
            self.tfBcaster.sendTransform(tf_msg)




### Markers stolen from pat3
class MarkerArrayPublisher:
    def __init__(self, topic, meshes, colors=[[0.2, 1., 0.2, 0.5]], scales=[(1., 1., 1.)], frame_id="w_ned"):
        self.meshes = meshes
        self.pub = rospy.Publisher(topic, visualization_msgs.msg.MarkerArray, queue_size=1)
        self.msg = visualization_msgs.msg.MarkerArray()
        for i, (mesh, color, scale) in enumerate(zip(meshes, colors, scales)):
            marker = visualization_msgs.msg.Marker()
            marker.header.frame_id = frame_id
            marker.type = marker.MESH_RESOURCE
            marker.action = marker.ADD
            marker.id = i
            #marker.text = "{}".format(i)
            marker.scale.x, marker.scale.y, marker.scale.z = scale
            marker.color.r, marker.color.g, marker.color.b, marker.color.a  = color
            #p = marker.pose.position; p.x, p.y, p.z = 0.3, 0, 0
            marker.mesh_resource = mesh
            marker.mesh_use_embedded_materials = True
            self.msg.markers.append(marker)
        
    def publish(self, T_ned2bs, delete=False):
        for marker, T_ned2b in zip(self.msg.markers, T_ned2bs):
            marker.action = marker.DELETE if delete else marker.ADD
            _position_and_orientation_from_T(marker.pose.position, marker.pose.orientation, T_ned2b)
        self.pub.publish(self.msg)

class PoseArrayPublisher(MarkerArrayPublisher):
    def __init__(self, topic='/pat/vehicle_marker', dae='quad.dae', scales=[(1., 1., 1.)], frame_id="w_ned"):
        MarkerArrayPublisher.__init__(self, topic,  ["package://ros_pat/media/{}".format(dae)], scales=scales, frame_id=frame_id)



#
#
#
class TwistSubscriber(SimpleSubscriber):
    def __init__(self, topic='cmd_vel', what='unkown', timeout=0.1, user_callback=None):
        SimpleSubscriber.__init__(self, topic, geometry_msgs.msg.Twist, what, timeout, user_callback)

    def get(self):
        msg = SimpleSubscriber.get(self)
        return msg.linear.x, msg.angular.z 


        
#
# Lanes
# 
class LaneModelPublisher(SimplePublisher):
    def __init__(self, topic, who='N/A'):
        SimplePublisher.__init__(self, topic, common_vision.msg.LaneModel, who)

    def publish(self, lm):
        msg = common_vision.msg.LaneModel()
        msg.header.stamp = lm.stamp
        msg.poly = lm.coefs
        msg.x_min = lm.x_min
        msg.x_max = lm.x_max
        SimplePublisher.publish(self, msg)

        
class LaneModelSubscriber(SimpleSubscriber):
    def __init__(self, topic, what='', timeout=0.1, user_cbk=None):
        SimpleSubscriber.__init__(self, topic, common_vision.msg.LaneModel, what, timeout, user_cbk)

    def get(self, lm):
        msg = SimpleSubscriber.get(self) # raise exceptions
        lm.coefs = self.msg.poly
        lm.x_min = self.msg.x_min
        lm.x_max = self.msg.x_max
        lm.stamp = self.msg.header.stamp
        lm.set_valid(True)

#
# Guidance Status
#
import two_d_guidance.msg # FIXME: cleary not!!!!
class GuidanceStatusPublisher(SimplePublisher):
    def __init__(self, topic='guidance/status', what='N/A', timeout=0.1, user_callback=None):
        SimplePublisher.__init__(self, topic, two_d_guidance.msg.FLGuidanceStatus, what) # FIXME trr.msg.GuidanceStatus

    def publish(self, model):
        msg = two_d_guidance.msg.FLGuidanceStatus()
        msg.guidance_mode = model.mode
        msg.poly = model.lane.coefs
        msg.x_min = model.lane.x_min
        msg.x_max = model.lane.x_max
        msg.lookahead_dist = model.lookahead_dist
        msg.lookahead_time = model.lookahead_time
        msg.carrot_x, msg.carrot_y = model.carrot
        msg.R = model.R
        msg.lin_sp, msg.ang_sp = model.lin_sp, model.ang_sp
        SimplePublisher.publish(self, msg)

class GuidanceStatusSubscriber(SimpleSubscriber):
    def __init__(self, topic='trr_guidance/status', what='N/A', timeout=0.1, user_callback=None):
        SimpleSubscriber.__init__(self, topic, two_d_guidance.msg.FLGuidanceStatus, what, timeout, user_callback)
        
    # def get(self):
    #     msg = SimpleSubscriber.get(self) # raise exceptions
    #     return msg

        
#
# Algebra/Transforms, stolen from pat3
#


def _position_and_orientation_from_T(p, q, T):
    p.x, p.y, p.z = T[:3, 3]
    q.x, q.y, q.z, q.w = tf.transformations.quaternion_from_matrix(T)

# Transforms
def T_of_t_rpy(t, rpy):
    T = tf.transformations.euler_matrix(rpy[0], rpy[1], rpy[2], 'sxyz')
    T[:3,3] = t
    return T

def t_rpy_of_T(T):
    t = T[:3,3]
    rpy = tf.transformations.euler_from_matrix(T, 'sxyz')
    return t, rpy

def T_of_t_q(t, q):
    T = tf.transformations.quaternion_matrix(q)
    T[:3,3] = t
    return T

def T_of_t_R(t, R):
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
    return T

def T_of_t_r(t, r):
    R, _ = cv2.Rodrigues(r)
    return T_of_t_R(t, R)

def tq_of_T(T):
    return T[:3, 3], tf.transformations.quaternion_from_matrix(T)

def tR_of_T(T):
    return T[:3,3], T[:3,:3]

def tr_of_T(T):
    ''' return translation and rodrigues angles from a 4x4 transform matrix '''
    r, _ = cv2.Rodrigues(T[:3,:3])
    return T[:3,3], r.squeeze()


def transform(a_to_b_T, p_a):
    return np.dot(a_to_b_T[:3,:3], p_a) + a_to_b_T[:3,3]

            
####
## Skeletons for common nodes

# Node with a periodic callback
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

# This node will fetch a camera, instantiate a pipeline, passing it the fetched camera    
class SimpleVisionPipeNode:
    def __init__(self, pipeline_class, pipe_cbk=None, pipe_args=None, img_fmt="passthrough", fetch_extrinsics=True):
        robot_name = rospy.get_param('~robot_name', '')
        def prefix(robot_name, what): return what if robot_name == '' else '{}/{}'.format(robot_name, what)
        self.cam_name = rospy.get_param('~camera', prefix(robot_name, 'camera_road_front'))
        self.ref_frame = rospy.get_param('~ref_frame', prefix(robot_name, 'base_link_footprint'))

        self.cam = retrieve_cam(self.cam_name, fetch_extrinsics=fetch_extrinsics, world=self.ref_frame)
        self.cam.set_undistortion_param(alpha=1.)

        self.cam_lst = CameraListener(self.cam_name, self.on_image, img_fmt)
        if pipe_args is not None:
            self.pipeline = pipeline_class(self.cam, robot_name, **pipe_args)
        else:
            self.pipeline = pipeline_class(self.cam, robot_name)


        
    def start(self):
        self.cam_lst.start()

    def started(self): return self.cam_lst.started()
    
    def stop(self):
        self.cam_lst.unregister()
        
    # we get a bgr8 image as input
    def on_image(self, img_bgr, arg):
        stamp, seq = arg
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


