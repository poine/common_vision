#!/usr/bin/env python3

import sys, numpy as np, rospy
import cv2, cv2.aruco as aruco

import rospy
from std_msgs.msg import String

import common_vision.rospy_utils as cv_rpu
import common_vision.utils as cv_u
import common_vision.camera as cv_c


import pdb

MY_DICT = aruco.DICT_5X5_1000    
MY_BOARD_PARAMS_A4 = {'squaresX':7,
                      'squaresY':5,
                      'squareLength':0.04,
                      'markerLength':0.03} # 5px/mm
class NonePipeline(cv_u.Pipeline):
    show_none = 0
    def __init__(self, cam, robot_name, bp=MY_BOARD_PARAMS_A4):
        self.cam = cam
        self.img = None
        cv_u.Pipeline.__init__(self)
        self.display_mode = 1

        self.aruco_dict = cv2.aruco.Dictionary_get(MY_DICT)
        self.bp = bp
        self.gridboard = aruco.CharucoBoard_create(**bp, dictionary=self.aruco_dict)
        self.corners, self.ids = [], []
        self.ref_2_cam_T = np.eye(4)
        
    def find_and_localize_board(self, img, cam):
        self.gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect markers
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(self.gray, self.aruco_dict)
        #print(f'found {len(ids)} markers')
        # refine markers detection using board description
        res = aruco.refineDetectedMarkers(self.gray, self.gridboard, corners, ids, rejectedImgPoints, cam.K, cam.D)
        detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = res
        #print(f'improved to {len(detectedIds)} markers')
        #self.img = aruco.drawDetectedMarkers(image=self.img, corners=corners)
        self.img = aruco.drawDetectedMarkers(image=self.img, corners=detectedCorners)
        #print(f'dmarkers {ids.squeeze()} refmarkers {detectedIds.squeeze()}')
        # Get charuco corners and ids from detected aruco markers
        response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=detectedCorners,
            markerIds=detectedIds,
            image=self.gray,
            board=self.gridboard)
        #print(f'charuco points {response}')
        #self.img = aruco.drawDetectedCornersCharuco(image=self.img,charucoCorners=charuco_corners,charucoIds=charuco_ids)

        ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.gridboard,
                                                             cam.K, cam.D, None, None, False)
        #print(ret, rvec, tvec)
        return self.gray, ret, rvec, tvec

    def unwarp_board(self, img, cam, rvec, tvec):
            corners_img, jac = cv2.projectPoints(self.gridboard.chessboardCorners, rvec, tvec, cam.K, cam.D)
            corners_undist = cam.undistort_points(corners_img)

            img_max_dim=1000
            bw, bh = self.bp['squaresX']*self.bp['squareLength'], self.bp['squaresY']*self.bp['squareLength']
            scale = img_max_dim/max((bw, bh))
            unwarped_size = (np.rint(np.array([bw, bh])*scale)).astype(int)
            #import pdb; pdb.set_trace()   
            H, status = cv2.findHomography(srcPoints=corners_undist, dstPoints=self.gridboard.chessboardCorners*scale,
                                   method=cv2.RANSAC, ransacReprojThreshold=0.01)
            self.img_undist = cam.undistort_img(self.img)
            self.img_unwarped = cv2.warpPerspective(self.img_undist, H, tuple(unwarped_size), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            self.img_unwarped = cv2.flip(self.img_unwarped, 1)

    def compute_cam_extrinsics(self, img, cam, rvec, tvec):
        board_2_cam_T = cv_u.T_of_t_r(tvec.squeeze(), rvec)
        #pdb.set_trace()
        if 1:
            cam.set_pose_T(board_2_cam_T)
            pt_board = np.array([[0,0,0], [0.16, 0, 0], [0, 0.16, 0], [0.16, 0.16, 0]], dtype=float)
            pts_img = cam.project(pt_board)
            for pt in pts_img[:,0]:
                cv2.circle(img, tuple(pt.astype(int)), 2, (0,255,0), -1)
        if 1:
            #pt_ref = np.array([[0., 0., 0.]], dtype=float)
            #pt_ref = np.array([[0.23,0.02, 0.]], dtype=float)
            #pt_ref = np.array([[0.415,0.02, 0.005]], dtype=float)
            # where you placed the chessboard wrt the robot
            #self.ref_2_board_t = [0.14, 0.51, -0.005] # ref (base link footprint) to ar board transform
            #self.ref_2_board_R = [[0, -1, 0],[-1, 0, 0],[0, 0, 1]]
            self.ref_2_board_t = [0.430, 0.1, -0.005] # ref (base link footprint) to ar board transform
            self.ref_2_board_R = [[-1, 0, 0],[0, -1, 0],[0, 0, 1]]
            #print(np.linalg.det(self.ref_2_board_R))
            pt_ref = self.ref_2_board_t + np.array([[0,-0.1,0.01], [-0.08, -0.1, 0.01], [-0.16, -0.1, 0.01], [-0.24, -0.1, 0.01],
                               [-0.16, -0.1, 0.01], [-0.16, -0.1, 0.01], [-0.24, -0.14, 0.01]], dtype=float)



            
            ref_2_board_T = cv_u.T_of_t_R(self.ref_2_board_t, self.ref_2_board_R)
            if 0: # good
                pt_board = np.array([cv_u.transform(ref_2_board_T, _pr) for _pr in pt_ref])
                print(pt_board)
                cam.set_pose_T(board_2_cam_T)
                pts_img = cam.project(pt_board)
            else: # good
                self.ref_2_cam_T = np.dot(board_2_cam_T, ref_2_board_T)
                cam.set_pose_T(self.ref_2_cam_T)
                pts_img = cam.project(pt_ref)

            f, h, c, w = cv_u.get_default_cv_text_params()
            h=1.
            cv2.putText(img, f'{pt_ref[0]}', tuple(pts_img[0,0].astype(int)), f, h, c, w)
            #print(pts_img.squeeze())
            for pt in pts_img[:,0]:
                cv2.circle(img, tuple(pt.astype(int)), 3, (255,255,0), -1)

    
    def _process_image(self, img, cam, stamp):
        self.img = img
        gray, ret, rvec, tvec = self.find_and_localize_board(img, cam)
        if ret:
            cv2.aruco.drawAxis(self.img, cam.K, cam.D, rvec, tvec, 0.12)
            self.compute_cam_extrinsics(img, cam, rvec, tvec)
            self.unwarp_board(img, cam, rvec, tvec)


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
            #if self.ids is not None and len(self.ids)>0:
            #    cv2.aruco.drawDetectedMarkers(debug_img, self.corners, self.ids)
            self.draw_timing(debug_img)
            return debug_img

class Node(cv_rpu.SimpleVisionPipeNode):

    def __init__(self):
        # will load a camera (~camera and ~ref_frame)
        cv_rpu.SimpleVisionPipeNode.__init__(self, NonePipeline, self.pipe_cbk, fetch_extrinsics=False)
        self.img_pub = cv_rpu.ImgPublisher(self.cam, '/vision/calibrate_extr')
        # rostopic pub /vision/calibrate_extr/save_to_disk std_msgs/String "/tmp/extcalib.yaml"
        self.msg_sub = rospy.Subscriber('/vision/calibrate_extr/save_to_disk', String, self.msg_cbk)
        self.start()

    def msg_cbk(self, data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
        world_to_cam_t, world_to_cam_q = cv_u.tq_of_T(self.pipeline.ref_2_cam_T)
        print(' world_to_cam_t {} world_to_cam_q {}'.format(world_to_cam_t, world_to_cam_q))
        filename = data.data#'/tmp/foo.yaml'
        cv_c.write_extrinsics(filename, world_to_cam_t, world_to_cam_q)
        print(f' saved calib to {filename}')
        
    def pipe_cbk(self):
        pass

    def periodic(self):
        if self.pipeline.display_mode != self.pipeline.show_none:
            self.img_pub.publish(self.pipeline, self.cam)
        #print(self.pipeline.ref_2_cam_T)
        world_to_cam_t, world_to_cam_q = cv_u.tq_of_T(self.pipeline.ref_2_cam_T)
        print(' world_to_cam_t {} world_to_cam_q {}'.format(world_to_cam_t, world_to_cam_q))
        #cv_c.write_extrinsics('/tmp/foo.yaml', world_to_cam_t, world_to_cam_q)

            
    def draw_debug(self, cam, img_cam=None):
        return self.pipeline.img

        
def main(args):
    name = 'vision_calibrate_extrinsic_node'
    rospy.init_node(name)
    rospy.loginfo('{} starting'.format(name))
    rospy.loginfo('  using opencv version {}'.format(cv2.__version__))
    Node().run(low_freq=2)


if __name__ == '__main__':
    main(sys.argv)
