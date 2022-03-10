#!/usr/bin/env python3

import os, numpy as np
import pdb
import common_vision.camera as cv_c
import common_vision.utils as cv_u

#/opt/ros/noetic/lib/tf2_ros/static_transform_publisher 0.224734856988 0.430927372545 1.928216639440 0.975193933310 0.004436339122 -0.027109297475 0.219641064741 base_link_footprint camera1_optical_frame __name:=cam1_optical_frame_to_world_frame_publisher


#prog_path = '/opt/ros/noetic/lib/tf2_ros/static_transform_publisher'
prog_path = '/opt/ros/noetic/lib/tf/static_transform_publisher'
#extr_path = '/home/ubuntu/work/trilosaurus/trilosaurus_bringup/cfg/camera1_extrinsics.yaml'
extr_path = '/home/ubuntu/work/robot_data/trilopi/camera1_extrinsics.yaml'
ref_to_camo_t, ref_to_camo_q = cv_c.load_extrinsics(extr_path, verbose=False)
print(f'ref_to_camo {ref_to_camo_t} {ref_to_camo_q}')
ref_to_camo_T = cv_u.T_of_t_q(ref_to_camo_t, ref_to_camo_q)
camo_to_ref_T = np.linalg.inv(ref_to_camo_T)
camo_to_ref_t, camo_to_ref_q = cv_u.tq_of_T(camo_to_ref_T)
if 1:
    _t, _q = camo_to_ref_t, camo_to_ref_q
    #_q = [1, 0, 0, 0]
else:
    _t, _q = ref_to_camo_t, ref_to_camo_q
args = [ str(_x) for _x in _t]+[str(_x) for _x in _q]
args += ['base_link_footprint', 'camera1_optical_frame',  '100', '__name:=cam1_optical_frame_to_world_frame_publisher']
#pdb.set_trace()
#args =  ['0.224734856988', '0.430927372545', '1.928216639440', '0.975193933310', '0.004436339122', '-0.027109297475', '0.219641064741', 'base_link_footprint', 'camera1_optical_frame',  '__name:=cam1_optical_frame_to_world_frame_publisher']


os.execlp(prog_path, *args)
