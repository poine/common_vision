#!/usr/bin/env python3

#
# Let's try to calibrate extrinsics on camera mounted vehicles
#


import sys, os, glob, yaml, cv2, numpy as np, matplotlib.pyplot as plt

import cv2, cv2.aruco as aruco
import pdb
import common_vision.utils as cv_u
import common_vision.camera as cv_c

MY_DICT = aruco.DICT_5X5_1000
MY_BOARD_PARAMS_A4 = {'squaresX':7,
                      'squaresY':5,
                      'squareLength':0.04,
                      'markerLength':0.03} # 5px/mm
MY_BOARD_PARAMS_A3 = {'squaresX':7,
                      'squaresY':5,
                      'squareLength':0.08,
                      'markerLength':0.06} # 7px/mm

def generate_chessboard(bp=MY_BOARD_PARAMS_A4, filename='/tmp/my_charuco_board.png', res=5000):
    # Create ChArUco board (set of Aruco markers in a chessboard setting meant for calibration)
    ar_dict = aruco.Dictionary_get(MY_DICT)
    gridboard = aruco.CharucoBoard_create(**bp, dictionary=ar_dict)
    bw, bh = bp['squaresX']*bp['squareLength'], bp['squaresY']*bp['squareLength']
    print(f'board size: {bw} x {bh} m')
    im_w, im_h = im_s = (int(bw*res), int(bh*res))
    img = gridboard.draw(outSize=im_s)
    cv2.imwrite(filename, img)
    print(f'board size: {im_w} x {im_h} px')
    # Display the image to us
    #cv2.imshow('board', img)
    cv_u.imshow_scaled(img, 'board')
    

    
# https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/sandbox/ludovic/aruco_calibration_rotation.html
# https://github.com/kyle-bersani/opencv-examples/blob/master/CalibrationByCharucoBoard/CalibrateCamera.py
def calib_intr(img_filenames, bp=MY_BOARD_PARAMS_A4):
    imgs = [cv2.imread(filename) for filename in img_filenames]
    ar_dict = aruco.Dictionary_get(MY_DICT)
    gridboard = aruco.CharucoBoard_create(**bp, dictionary=ar_dict)
    corners_all, ids_all = [], [] # Corners and ids discovered in all images processed
    image_size = None # Determined at runtime
    for img in imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None: image_size = gray.shape[::-1]
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, ar_dict)
        print(f'found {len(ids)} markers')
        img = aruco.drawDetectedMarkers(image=img, corners=corners)
        # Get charuco corners and ids from detected aruco markers
        response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=gridboard)
        print(response)
        if response >= 20:
            corners_all.append(charuco_corners)
            ids_all.append(charuco_ids)
            # Draw the Charuco board we've detected
            img = aruco.drawDetectedCornersCharuco(
                image=img,
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids)
    rep_err, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=gridboard,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None)
    
    # Print matrix and distortion coefficient to the console
    print(rep_err)
    print(cameraMatrix)
    print(distCoeffs)

    cv_u.imshow_mosaic(imgs, 'board1')
    cv_c.write_intrinsics2('/tmp/foo.yaml', image_size, cameraMatrix, distCoeffs, a=1., cname='unknown')


def calib_extr():
    pass

    
if __name__ == '__main__':
    #in_filename =  '/home/poine/work/smocap/smocap/test/ricou/calib_floor_cam.png'
    #intr_cam_calib_path = '/home/poine/work/smocap/smocap/params/enac_demo_z/ueye_enac_z_2.yaml'
    #extr_cam_calib_path = '/home/poine/work/smocap/smocap/params/ricou/ueye_enac_z2_extr.yaml'
    #generate_chessboard(bp=MY_BOARD_PARAMS_A4, filename='/tmp/my_charuco_board_a4.png', res=5000)
    #generate_chessboard(bp=MY_BOARD_PARAMS_A3, filename='/tmp/my_charuco_board_a3.png', res=5000)
    img_filenames = [f'/home/poine/charuco_board_a4_view_{i}.png' for i in range(1,4)]
    calib_intr(img_filenames)
    # Exit on any key
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
