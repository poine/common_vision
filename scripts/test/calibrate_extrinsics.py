#!/usr/bin/env python

import logging, glob, yaml
import numpy as np, cv2
import matplotlib.pyplot as plt
import tf.transformations

import common_vision.utils as cv_u
import common_vision.plot_utils as cv_pu
import common_vision.camera as cv_c
#import utils

import pdb

LOG = logging.getLogger('calibrate_extrinsic')
logging.basicConfig(level=logging.INFO)


def calibrate_extrinsics(cam_intrinsics_path, points_path):

    # load camera intrinsics
    cam = cv_c.load_cam_from_files(cam_intrinsics_path)

    #camera_matrix, dist_coeffs, w, h = utils.load_camera_model(cam_intrinsics_path)
    # load keypoints
    pts_name, pts_img, pts_world = cv_u.read_point(points_path)
    # run PnP to obtain camera pose
    (success, rotation_vector, translation_vector) = cv2.solvePnP(pts_world, pts_img.reshape(-1, 1, 2),
                                                                  cam.K, cam.D, flags=cv2.SOLVEPNP_ITERATIVE)
    LOG.info("PnP {} rotation {} translation {}".format(success, rotation_vector.squeeze(), translation_vector.squeeze()))
    # compute reprojection and reprojection error
    rep_pts_img =  cv2.projectPoints(pts_world, rotation_vector, translation_vector, cam.K, cam.D)[0].squeeze()
    rep_err = np.mean(np.linalg.norm(pts_img - rep_pts_img, axis=1))
    LOG.info('reprojection error {} px'.format(rep_err))
    return rotation_vector, translation_vector, pts_name, pts_img, pts_world, rep_pts_img


def draw_result_image(img_path, pts_name, pts_img, pts_world, rep_pts_img):
    # load the floor tile image
    img = cv2.imread(img_path,  cv2.IMREAD_UNCHANGED)
    # Draw keypoint on image, write their names and coordinates
    for i, p in enumerate(pts_img.astype(int)):
        cv2.circle(img, tuple(p), 1, (0,255,0), -1)
        cv2.putText(img, '{}'.format(pts_name[i][:1]), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.putText(img, '{}'.format(pts_world[i][:2]), tuple(p+[0, 25]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    for i, p in enumerate(rep_pts_img.astype(int)):
        cv2.circle(img, tuple(p), 1, (0,0,255), -1)
    cv2.imshow('original image', img)
    #cv2.imshow('undistorted image', img_undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_result_3D(ref_to_cam_T, pts_world=None, pts_name=None):
    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')
    cv_pu.draw_thriedra(ax, np.eye(4), id='local floor plane', scale=0.2)
    cv_pu.draw_thriedra(ax, np.linalg.inv(ref_to_cam_T), id='Camera', scale=0.2)
    #ax.scatter(pts_world[:,0], pts_world[:,1], pts_world[:,2], marker='o')
    if pts_world is not None:
        cv_pu.draw_points(ax, pts_world, pts_name)
    cv_pu.set_3D_axes_equal()

    plt.show()

def write_yaml(filename, ref_to_camo_T, comment=None):
    LOG.info("saving calibration to {}".format(filename))
    ref_to_camo_t, ref_to_camo_q = utils.tq_of_T(ref_to_camo_T)
    with open(filename, 'w') as f:
        if comment is not None:  f.write('# {}\n'.format(comment))
        f.write('ref_to_camo_t: {}\n'.format(", ".join(["{:.8f}".format(ref_to_camo_t[i]) for i in range(len(ref_to_camo_t))])))
        f.write('ref_to_camo_q: {}\n'.format(", ".join(["{:.8f}".format(ref_to_camo_q[i]) for i in range(len(ref_to_camo_q))])))
    


def test_caml_to_camo():
    '''
    From robot URDF, we specify the camera_link to ref transform.
    <xacro:property name="cam_cl_to_ref_xyz" value="0.03418817 -0.00488509  0.17943348" />
    <xacro:property name="cam_cl_to_ref_rpy" value="0.00211058  0.64585395  0.03289786" />
    '''
    T_cl2co = np.array([[  0.,  -1.,  0. , 0],
                        [  0.,   0., -1. , 0],
                        [  1.,   0.,  0. , 0],
                        [  0.,   0.,  0. , 1.]])
    
    cam_cl_to_ref_xyz = [0.03418817, -0.00488509,  0.17943348]
    cam_cl_to_ref_rpy = [0.00211058,  0.64585395,  0.03289786]
    cl_to_ref_T = tf.transformations.euler_matrix(axes='sxyz', *cam_cl_to_ref_rpy)
    cl_to_ref_T[:3,3] = cam_cl_to_ref_xyz
    ref_to_cl_T = np.linalg.inv(cl_to_ref_T)
    ref_to_co_T = np.dot(T_cl2co, ref_to_cl_T)
    #draw_result_3D(ref_to_cl_T)
    #draw_result_3D(ref_to_co_T)
    write_yaml('data/caroline_gazebo/caroline_camera_road_extr2.yaml', ref_to_co_T, 'extrinsics (base footprint to camera optical frame transform)')

def print_caml_transforms(ref_to_camo_T):
    caml_to_camo_T = np.array([[  0.,  -1.,  0. , 0],
                               [  0.,   0., -1. , 0],
                               [  1.,   0.,  0. , 0],
                               [  0.,   0.,  0. , 1.]])
    camo_to_caml_T = np.linalg.inv(caml_to_camo_T)
    ref_to_caml_T = np.dot(camo_to_caml_T, ref_to_camo_T)
    caml_to_ref_T = np.linalg.inv(ref_to_caml_T)
    rpy = np.array(tf.transformations.euler_from_matrix(caml_to_ref_T, 'sxyz'))
    #pdb.set_trace()
    xyz = caml_to_ref_T[:3,3]
    print( 'rpy {}'.format(rpy))
    print( 'xyz {}'.format(xyz))

def main():
    pass
        
def test_christine():
    # compute extrinsics
    rotation_vector, translation_vector, pts_name, pts_img, pts_world, rep_pts_img = calibrate_extrinsics('data/christine_z/christine_camera_road_front.yaml', 'data/christine_z/pts_floor_tiles_z_04.yaml')
    world_to_camo_T = utils.T_of_tr(translation_vector.squeeze(), rotation_vector)
    write_yaml('data/christine_z/christine_camera_road_front_extr2.yaml', world_to_camo_T, 'extrinsics (base footprint to camera optical frame transform)')
    #draw_result_image('data/christine_z/floor_tiles_z_04.png', pts_name, pts_img, pts_world, rep_pts_img)
    draw_result_3D(world_to_camo_T, pts_world, pts_name)
    
def test_caroline():
    intr_file = '/home/poine/.ros/camera_info/caroline_camera_one.yaml'
    img_file = '/home/poine/work/robot_data/caroline/floor_tiles_ricou_01.png'
    calib_pts_file = '/home/poine/work/roverboard/roverboard_caroline/config/ext_calib_pts_floor_tiles_ricou_01.yaml'
    LOG.info("Loading image {}".format(img_file))
    LOG.info("Loading pts {}".format(calib_pts_file))
    # compute extrinsics
    rotation_vector, translation_vector, pts_name, pts_img, pts_world, rep_pts_img = calibrate_extrinsics(intr_file, calib_pts_file)
    world_to_camo_T = cv_u.T_of_t_r(translation_vector.squeeze(), rotation_vector)
    print_caml_transforms(world_to_camo_T)
    #write_yaml('data/caroline_gazebo/caroline_camera_road_extr2.yaml', world_to_camo_T, 'extrinsics (base footprint to camera optical frame transform)')
    draw_result_image(img_file, pts_name, pts_img, pts_world, rep_pts_img)
    #draw_result_3D(world_to_camo_T, pts_world, pts_name)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    #test_caml_to_camo()
    test_caroline()
    #test_christine()
