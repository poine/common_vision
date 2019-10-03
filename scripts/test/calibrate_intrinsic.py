#!/usr/bin/env python
import numpy as np, cv2, matplotlib.pyplot as plt
import logging
LOG = logging.getLogger('calibrate_intrisinc')
import common_vision.utils as cvu
import common_vision.plot_utils as cvpu
import common_vision.camera as cvc

'''
Detect chessboards in all images
'''
def detect_chessboards(images, imgs_path, cb_geom=(8,6), cb_size=0.108, refine_corners=False):
    img_points, object_points, rets = [], [], [] # 2d points in image, 3d point in real world space, detection_status
    cb_points = np.zeros((cb_geom[0]*cb_geom[1], 3), np.float32)
    cb_points[:,:2] = cb_size*np.mgrid[0:cb_geom[0],0:cb_geom[1]].T.reshape(-1,2)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_FILTER_QUADS|cv2.CALIB_CB_SYMMETRIC_GRID

    for img_gray, _path in zip(images, imgs_path):
        ret, corners = cv2.findChessboardCorners(img_gray, cb_geom, flags=flags)
        rets.append(ret)
        if ret:
            object_points.append(cb_points)
            if refine_corners:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                #corners = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)
                corners = cv2.cornerSubPix(img_gray, corners, (5,5), (-1,-1), criteria)
            img_points.append(corners)
        else:
            LOG.info(' chessboard detection failed for {}'.format(_path))
    LOG.info(" successfully extracted {} chessboard pattern in {} images".format(cb_geom, len(img_points)))
    return img_points, object_points, img_gray.shape[::-1], rets

'''
Run opencv camera intrinsic calibration
'''
def calibrate_camera(img_points, object_points, img_shape, rational_model=False):
    flags = cv2.CALIB_RATIONAL_MODEL if rational_model else 0
    flags |= cv2.CALIB_FIX_K3
    rep_err, cmtx, distk, rvecs, tvecs = cv2.calibrateCamera(object_points, img_points, img_shape, None, None, flags=flags)
    LOG.info(" cv calibration:")
    LOG.info("   Reprojection error: {:.3f} pixels".format(rep_err))
    LOG.info("   Camera matrix:\n{}".format(cmtx))
    LOG.info("   Distortion coeffs:\n{}".format(distk))
    return  cmtx, distk, rvecs, tvecs



'''

'''
def report_calibration(imgs_path, rep_errs):
    for _p, _re in zip(imgs_path, rep_errs):
        LOG.info("img: {} re(mean/min/max): {:4.2f} {:4.2f} {:4.2f} pixels".format(_p, np.mean(_re), np.min(_re), np.max(_re)))
    

'''
  Runs opencv intrinsic calibration on a batch of images
  Should be equivalent to:
    /opt/ros/melodic/lib/camera_calibration/tarfile_calibration.py --mono --visualize -q 0.025 -s 8x6 /tmp/enac_drone_july_4_2019.tgz
'''
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    LOG.info(" using opencv version: {}".format(cv2.__version__))

    #_dir, _img_prefix = '/mnt/mint17/home/poine/work/teaching_material/automatique/trunk/vision/simulations/camera_calibration/poine_pixel/', 'IMG_'
    _dir, _img_prefix = '/mnt/mint17/home/poine/work/teaching_material/automatique/trunk/vision/simulations/camera_calibration/ueye_poine_front/2016_03_27', 'left'
    #_dir, _img_prefix = '/tmp/cam_road_christine', 'left-'
    imgs, imgs_path = cvu.load_images_in_dir(_dir, _img_prefix)

    #imgs, imgs_path = cvu.load_images_in_tarfile('/tmp/foo.tgz', _img_prefix)
    #imgs, imgs_path = cvu.load_images_in_tarfile('/home/poine/work/cameras/ueye_poine_3_2019_september_30.tgz', _img_prefix)
     
    img_points, object_points, img_shape, rets = detect_chessboards(imgs, imgs_path, cb_size=0.025, refine_corners=False)
    cmtx, distk, rvecs, tvecs = calibrate_camera(img_points, object_points, img_shape)
    
    rep_pts, rep_errs = cvu.compute_reprojection_error(img_points, object_points, cmtx, distk, rvecs, tvecs)

    report_calibration(imgs_path, rep_errs)
    
    filename = '/tmp/foo.yaml'
    LOG.info(" saving to :{}".format(filename))
    cvc.write_intrinsics2(filename, img_shape, cmtx, distk, a=1., cname='unknown')


    
    #cvpu.plot_images(imgs, imgs_path, img_points, rets, rep_pts)
    cvpu.plot_images2(imgs, imgs_path, img_points, rets, rep_pts)
    plt.show()
