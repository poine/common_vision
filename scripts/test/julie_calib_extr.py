#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging, glob, yaml

LOG = logging.getLogger('calibrate_extrinsic')
logging.basicConfig(level=logging.INFO)

import common_vision.utils as cv_u
  
import calibrate_extrinsics as calib_ext



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    intr_file = '/home/poine/.ros/camera_info/julie_short_range.yaml'
    img_file = '/home/poine/work/robot_data/julie/julie_extr_calib_1.png'
    calib_pts_file = '/home/poine/work/robot_data/julie/ext_calib_pts_1.yaml'
    LOG.info("Loading image {}".format(img_file))
    LOG.info("Loading pts {}".format(calib_pts_file))
    # compute extrinsics
    rotation_vector, translation_vector, pts_name, pts_img, pts_world, rep_pts_img =  calib_ext.calibrate_extrinsics(intr_file, calib_pts_file)
    world_to_camo_T = cv_u.T_of_t_r(translation_vector.squeeze(), rotation_vector)
    calib_ext.print_caml_transforms(world_to_camo_T)
    calib_ext.write_yaml('/home/poine/work/robot_data/julie/julie_short_range_extr.yaml', world_to_camo_T,
                         'extrinsics (base footprint to camera optical frame transform)')
    calib_ext.draw_result_image(img_file, pts_name, pts_img, pts_world, rep_pts_img)
    calib_ext.draw_result_3D(world_to_camo_T, pts_world, pts_name)
