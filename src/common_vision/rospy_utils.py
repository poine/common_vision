import numpy as np
import rospy, tf2_ros, sensor_msgs.msg

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
