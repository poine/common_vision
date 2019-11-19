#!/usr/bin/env python
import os, logging, time, math, numpy as np
from matplotlib import pyplot as plt

import cv2, yaml
import shapely, shapely.geometry
import pdb

LOG = logging.getLogger('bird_eye')

def _make_line(p0, p1, spacing=1, endpoint=True):
    dist = np.linalg.norm(p1-p0)
    n_pt = dist/spacing
    if endpoint: n_pt += 1
    return np.stack([np.linspace(p0[j], p1[j], n_pt, endpoint=endpoint) for j in range(len(p0))], axis=-1)

def _lines_of_corners(corners, spacing):
    return np.concatenate([_make_line(corners[i-1], corners[i], spacing=spacing, endpoint=False) for i in range(len(corners))])

def get_points_on_plane(rays, plane_n, plane_d):
    return np.array([-plane_d/np.dot(ray, plane_n)*ray for ray in rays])


class BirdEye:

    def __init__(self, cam, param):
        self._set_param(cam, param)



    def _set_param(self, cam, param):
        self.param = param
        self.corners_world = np.array([(param.x0, param.dy/2, 0.), (param.x0+param.dx, param.dy/2, 0.),
                                       (param.x0+ param.dx, -param.dy/2, 0.), (param.x0, -param.dy/2, 0.)])
        self._compute_cam_viewing_area(cam)
        self._compute_H_world(cam)
        self._compute_H_unwarped(cam)
        self._compute_image_mask(cam)

    def _compute_cam_viewing_area(self, cam, max_dist=6):
        # Compute the contour of the intersection between camera frustum and floor plane (cliping to max_dist)
        cam_va_corners_img = np.array([[0., 0], [cam.w, 0], [cam.w, cam.h], [0, cam.h], [0, 0]])
        cam_va_borders_img = _lines_of_corners(cam_va_corners_img, spacing=1)
        cam_va_borders_undistorted = cam.undistort_points(cam_va_borders_img.reshape(-1, 1, 2))
        cam_va_borders_imp = np.array([np.dot(cam.inv_undist_K, [u, v, 1]) for (u, v) in cam_va_borders_undistorted.squeeze()])
        cam_va_borders_fp_cam = get_points_on_plane(cam_va_borders_imp, cam.fp_n, cam.fp_d)
        in_frustum_idx = np.logical_and(cam_va_borders_fp_cam[:,2]>0, cam_va_borders_fp_cam[:,2]<max_dist)
        cam_va_borders_fp_cam = cam_va_borders_fp_cam[in_frustum_idx,:]
        self.cam_va_borders_fp_world = np.array([np.dot(cam.cam_to_world_T[:3], p.tolist()+[1]) for p in cam_va_borders_fp_cam]) 

        # Compute intersection between camera viewing area and bird eye area
        poly_va_blf = shapely.geometry.Polygon(self.cam_va_borders_fp_world[:,:2])
        poly_be_blf = shapely.geometry.Polygon(self.corners_world[:,:2])
        _tmp = poly_va_blf.intersection(poly_be_blf).exterior.coords.xy
        self.borders_isect_be_cam_world = np.zeros((len(_tmp[0]), 3))
        self.borders_isect_be_cam_world[:,:2] = np.array(_tmp).T

    # Compute homography from image undistorted plan to base link footprint
    def _compute_H_world(self, cam):
        va_corners_img  = cam.project(self.borders_isect_be_cam_world)
        va_corners_imp  = cam.undistort_points(va_corners_img)
        self.H_world, status = cv2.findHomography(srcPoints=va_corners_imp, dstPoints=self.borders_isect_be_cam_world, method=cv2.RANSAC, ransacReprojThreshold=0.01)
        print('computed H blf: ({}/{} inliers)\n{}'.format(np.count_nonzero(status), len(va_corners_imp), self.H_world))

    # Compute homography from undistorted image plan to unwarped
    def _compute_H_unwarped(self, cam):
        va_corners_img  = cam.project(self.borders_isect_be_cam_world)
        va_corners_imp  = cam.undistort_points(va_corners_img)
        va_corners_unwarped = self.lfp_to_unwarped(cam, self.borders_isect_be_cam_world.squeeze())
        self.H_unwarped, status = cv2.findHomography(srcPoints=va_corners_imp, dstPoints=va_corners_unwarped, method=cv2.RANSAC, ransacReprojThreshold=0.01)
        print('computed H unwarped: ({}/{} inliers)\n{}'.format( np.count_nonzero(status), len(va_corners_imp), self.H_unwarped))

    # 
    def _compute_image_mask(self, cam):
        self.img_mask = cam.project(self.borders_isect_be_cam_world).squeeze()[np.newaxis].astype(np.int)
        

        
    # Convertions between lfp and unwarped     
    def lfp_to_unwarped(self, cam, cnt_lfp):
        cnt_uv = np.array([(self.param.w/2-_y/self.param.s, self.param.h-(_x-self.param.x0)/self.param.s) for _x, _y, _ in cnt_lfp])
        return cnt_uv

    def unwarped_to_fp(self, cam, cnt_uw):
        self.cnt_fp = np.array([((self.param.h-p[1])*self.param.s+self.param.x0, (self.param.w/2-p[0])*self.s, 0.) for p in cnt_uw.squeeze()])
        return self.cnt_fp

    # undistort, then unwarp image
    def undist_unwarp_img(self, img, cam):
        img_undist = cam.undistort_img(img)
        return cv2.warpPerspective(img_undist, self.H_unwarped, (self.param.w, self.param.h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)


    