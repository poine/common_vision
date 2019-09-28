#!/usr/bin/env python
import os, time, numpy as np, matplotlib.pyplot as plt, cv2
import rosbag, cv_bridge
import pdb


import common_vision.camera as cvc
import common_vision.lane.lane_4 as lane_4

def run_on_bag(pipe, cam, bag_path, img_topic, odom_topic, sleep=False, talk=False, display=True):
    bag, bridge = rosbag.Bag(bag_path, "r"), cv_bridge.CvBridge()
    durations, last_img_t = [], None
    pipe_times, lane_mod_coefs = [], []
    odom_times, odom_vlins, odom_vangs = [], [], []
    img_idx = -1
    for topic, msg, img_t in bag.read_messages(topics=[img_topic, odom_topic]):
        if topic == img_topic:
            img_idx += 1
            img_dt = 0.01 if last_img_t is None else (img_t-last_img_t).to_sec()
            cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            pipe.process_image(cv_img, cam, msg.header.stamp, msg.header.seq)
            lane_mod_coefs.append(pipe.lane_model.coefs)
            pipe_times.append(img_t.to_sec())
            durations.append(pipe.last_processing_duration)
            if talk: print('img {} {:.3f}s ({:.1f}hz)'.format(img_idx, pipe.last_processing_duration, 1./pipe.last_processing_duration ))
            if pipe.display_mode != pipe.show_none:
                out_img = pipe.draw_debug_bgr(cam)
                cv2.imshow('input', cv_img)
                cv2.imshow('pipe debug', out_img)
                cv2.waitKey(1)
                #cv2.waitKey(0)
            last_img_t = img_t
            time_to_sleep = max(0., img_dt-pipe.last_processing_duration) if sleep else 0
            time.sleep(time_to_sleep)
        elif topic == odom_topic:
            odom_times.append(msg.header.stamp.to_sec())
            odom_vlins.append(msg.twist.twist.linear.x)
            odom_vangs.append(msg.twist.twist.angular.z)
        
    freqs = 1./np.array(durations); _mean, _std, _min, _max = np.mean(freqs), np.std(freqs), np.min(freqs), np.max(freqs)
    plt.hist(freqs); plt.xlabel('hz'); plt.legend(['mean {:.1f} std {:.1f}\n min {:.1f} max {:.1f}'.format(_mean, _std, _min, _max)])

    lane_mod_coefs = np.array(lane_mod_coefs)
    pipe_times = np.array(pipe_times)

    filename = '/tmp/pipe_run.npz'
    print('saving run to {}'.format(filename))
    np.savez(filename, times = pipe_times, lane_mod_coefs=lane_mod_coefs,
             odom_times=odom_times, odom_vlins=odom_vlins, odom_vangs=odom_vangs)

    plot_run(pipe_times, lane_mod_coefs)

def plot_run(times, lane_mod_coefs):
    fig, axs = plt.subplots(4, 2)
    for i in range(4):
        axs[i, 0].plot(lane_mod_coefs[:,i])
        #axs[i, 1].plot(lane_mod_coefs[:-1,i] - lane_mod_coefs[1:,i])
    plt.show()

if __name__ == '__main__':
    robot_name = 'christine' 
    intr_cam_calib_path = '/home/poine/.ros/camera_info/{}_camera_road_front.yaml'.format(robot_name)
    extr_cam_calib_path = '/home/poine/work/oscar/oscar/oscar_description/cfg/{}_cam_road_front_extr.yaml'.format(robot_name)
    cam = cvc.load_cam_from_files(intr_cam_calib_path, extr_cam_calib_path)

    pipe = lane_4.Pipeline(cam, 'christine')
    pipe.display_mode = lane_4.Pipeline.show_summary

    bag_dir = '/home/poine'
    bag_filename = '2019-09-16-18-40-00.bag' # Christine Vedrines 1 tour, ombres longues, vitesse constante 2.5
    img_topic, odom_topic = '/camera_road_front/image_raw', '/oscar_ackermann_controller/odom'
    bag_path = os.path.join(bag_dir, bag_filename)
    run_on_bag(pipe, cam, bag_path, img_topic, odom_topic, sleep=True, talk=False)
