#!/usr/bin/env python
import os, logging, time, math, numpy as np
from matplotlib import pyplot as plt

import cv2, yaml
import shapely, shapely.geometry
import pdb

import common_vision.utils as cv_u

'''
 FIXME: tables seem to be missing, see BirdEyeTransformer in two_d_guidance/src/two_d_guidance/trr/vision/utils.py

Bird Eye View.
This object converts between camera image to the local floor plane (lfp). 
It is able to transform the camera image into a bird eye view, as if the camera was looking straight at the floor plane.
This transformed image is refered to as unwarped.
For efficiency purposes, transformation tables can be pre computed.
As this computation is expensive, tables can be loaded from filesystem.
'''

LOG = logging.getLogger('bird_eye')

def _make_line(p0, p1, spacing=1, endpoint=True):
    dist = np.linalg.norm(p1-p0)
    n_pt = int(dist/spacing)
    if endpoint: n_pt += 1
    return np.stack([np.linspace(p0[j], p1[j], n_pt, endpoint=endpoint) for j in range(len(p0))], axis=-1)

def _lines_of_corners(corners, spacing):
    return np.concatenate([_make_line(corners[i-1], corners[i], spacing=spacing, endpoint=False) for i in range(len(corners))])

def get_points_on_plane(rays, plane_n, plane_d):
    return np.array([-plane_d/np.dot(ray, plane_n)*ray for ray in rays])

class BirdEyeParam:
    def __init__(self, x0=0.3, dx=3., dy=2., w=640):
        # coordinates of viewing area on local floorplane in base_footprint frame
        self.x0, self.dx, self.dy = x0, dx, dy
        # coordinates of viewing area as a pixel array (unwarped)
        self.w = w; self.s = self.dy/self.w; self.h = int(self.dx/self.s)

        # viewing area in base_footprint frame
        # bottom_right, top_right, top_left, bottom_left in base_footprint frame
        self.corners_be_blf = np.array([(self.x0, self.dy/2, 0.), (self.x0+self.dx, self.dy/2, 0.), (self.x0+self.dx, -self.dy/2, 0.), (self.x0, -self.dy/2, 0.)])
        self.corners_be_img = np.array([[0, self.h], [0, 0], [self.w, 0], [self.w, self.h]])

class JulieBirdEyeParam(BirdEyeParam):
    def __init__(self, x0=2.7, dx=15., dy=8., w=640):
        BirdEyeParam.__init__(self, x0, dx, dy, w)
        
class BeParamTrilopi:
    #x0, y0, dx, dy = 0.11, 0., 0.25, 0.2 # bird eye area in local floor plane frame
    x0, y0, dx, dy = 0.10, 0., 0.15, 0.2 # bird eye area in local floor plane frame
    max_dist = 0.3
    w = 640                     # bird eye image width (pixel coordinates)
    s = dy/w                    # scale
    h = int(dx/s)               # bird eye image height


        
def NamedBirdEyeParam(_name):
    if    _name == 'caroline':  return CarolineBirdEyeParam()
    elif  _name == 'christine': return ChristineBirdEyeParam()
    elif  _name == 'caroline_jetson':  return CarolineJetsonBirdEyeParam()
    elif  _name == 'caroline_test_amcl':  return CarolineJetsonBirdEyeParam()
    elif  _name == 'julie':  return JulieBirdEyeParam()
    elif  _name == 'trilopi':  return BeParamTrilopi()
    return None


class BirdEye:

    def __init__(self, cam, param, cache_filename=None, force_recompute=False):
        self._set_param(cam, param, cache_filename, force_recompute)

    def _set_param(self, cam, param, cache_filename=None, force_recompute=False):
        self.param = param
        self.corners_lfp = np.array([(param.x0, param.y0+param.dy/2, 0.), (param.x0+param.dx, param.y0+param.dy/2, 0.),
                                       (param.x0+param.dx, param.y0-param.dy/2, 0.), (param.x0, param.y0-param.dy/2, 0.)])
        print(f'cache_filename {cache_filename}')
        if cache_filename is None or not os.path.exists(cache_filename) or force_recompute:
            try:
                self._compute_cam_viewing_area(cam, max_dist=param.max_dist)
                self._compute_H_lfp(cam)
                self._compute_H_unwarped(cam)
                self._compute_H_unwarped_map(cam)
                self._compute_undist_unwarp_map(cam)
                self._compute_image_mask(cam) # FIXME
            except shapely.errors.TopologicalError:
                self.borders_isect_be_cam_lfp = np.zeros((1, 3))
                print('miserable failure')
            except IndexError:
                print('miserable failure 2')
            else:
                if cache_filename is not None:
                    print(f'saving tables to {cache_filename}')
                    np.savez(cache_filename, H_lfp=self.H_lfp, unwarp_xmap=self.unwarp_xmap, unwarp_ymap=self.unwarp_ymap,
                             undist_unwarp_xmap=self.undist_unwarp_xmap_int,  undist_unwarp_ymap=self.undist_unwarp_ymap_int,
                             borders_isect_be_cam_lfp=self.borders_isect_be_cam_lfp,
                             unwarped_img_mask=self.unwarped_img_mask,
                             cam_img_mask=self.cam_img_mask)


        else:
            data =  np.load(cache_filename)
            print(f'loading precomputed data in {cache_filename}')
            self.H_lfp = data['H_lfp']
            self.unwarp_xmap = data['unwarp_xmap']
            self.unwarp_ymap = data['unwarp_ymap']
            self.undist_unwarp_xmap_int = data['undist_unwarp_xmap']
            self.undist_unwarp_ymap_int = data['undist_unwarp_ymap']
            self.borders_isect_be_cam_lfp = data['borders_isect_be_cam_lfp']
            self.unwarped_img_mask = data['unwarped_img_mask']
            self.cam_img_mask = data['cam_img_mask']
        self._compute_image_mask(cam)
            
    def _compute_cam_viewing_area(self, cam, max_dist):
        # Compute the contour of the intersection between camera's frustum and floor plane (cliping to max_dist)
        cam_va_corners_img = np.array([[0., 0], [cam.w, 0], [cam.w, cam.h], [0, cam.h]])#, [0, 0]])
        cam_va_borders_img = _lines_of_corners(cam_va_corners_img, spacing=1)
        cam_va_borders_undistorted = cam.undistort_points(cam_va_borders_img.reshape(-1, 1, 2))
        cam_va_borders_imp = np.array([np.dot(cam.inv_undist_K, [u, v, 1]) for (u, v) in cam_va_borders_undistorted.squeeze()])
        cam_va_borders_fp_cam = get_points_on_plane(cam_va_borders_imp, cam.fp_n, cam.fp_d)
        in_frustum_idx = np.logical_and(cam_va_borders_fp_cam[:,2]>0, cam_va_borders_fp_cam[:,2]<max_dist)
        cam_va_borders_fp_cam = cam_va_borders_fp_cam[in_frustum_idx,:]
        self.cam_va_borders_fp_lfp = np.array([np.dot(cam.cam_to_world_T[:3], p.tolist()+[1]) for p in cam_va_borders_fp_cam])
        print('computed cam viewing area on ground plane')
        #print(f'{self.cam_va_borders_fp_lfp}\n {self.corners_lfp}')
        
        # Compute intersection between camera viewing area and bird eye area
        poly_va_blf = shapely.geometry.Polygon(self.cam_va_borders_fp_lfp[:,:2])
        poly_be_blf = shapely.geometry.Polygon(self.corners_lfp[:,:2])
        #print(f'{poly_va_blf} {poly_be_blf}')
        _tmp = poly_va_blf.intersection(poly_be_blf).exterior.coords.xy
        self.borders_isect_be_cam_lfp = np.zeros((len(_tmp[0]), 3))
        self.borders_isect_be_cam_lfp[:,:2] = np.array(_tmp).T
        print('computed intersection of cam viewing area and bird eye area')

        
    # Compute homography from undistorted image plane to local floor plane
    def _compute_H_lfp(self, cam):
        va_corners_img  = cam.project(self.borders_isect_be_cam_lfp)
        va_corners_imp  = cam.undistort_points(va_corners_img)
        self.H_lfp, status = cv2.findHomography(srcPoints=va_corners_imp, dstPoints=self.borders_isect_be_cam_lfp, method=cv2.RANSAC, ransacReprojThreshold=0.01)
        print('computed H blf: ({}/{} inliers)\n{}'.format(np.count_nonzero(status), len(va_corners_imp), self.H_lfp))

    # Compute homography from undistorted image plane to unwarped
    def _compute_H_unwarped(self, cam):
        va_corners_img  = cam.project(self.borders_isect_be_cam_lfp)
        va_corners_imp  = cam.undistort_points(va_corners_img)
        va_corners_unwarped = self.lfp_to_unwarped(cam, self.borders_isect_be_cam_lfp.squeeze())
        self.H_unwarped, status = cv2.findHomography(srcPoints=va_corners_imp, dstPoints=va_corners_unwarped, method=cv2.RANSAC, ransacReprojThreshold=0.01)
        print('computed H unwarped: ({}/{} inliers)\n{}'.format( np.count_nonzero(status), len(va_corners_imp), self.H_unwarped))

    # Compute homography from undistorted image plane to unwarped as a map, to speed up computation
    def _compute_H_unwarped_map(self, cam):
        print('computing unwarp maps')
        self.unwarp_xmap = np.zeros((self.param.h, self.param.w), np.float32)
        self.unwarp_ymap = np.zeros((self.param.h, self.param.w), np.float32)
        Hinv = np.linalg.inv(self.H_unwarped)
        for y in range(self.param.h):
            for x in range(self.param.w):
                pt_be = np.array([[x], [y], [1]], dtype=np.float32)
                pt_imp = np.dot(Hinv, pt_be)
                pt_imp /= pt_imp[2]
                self.unwarp_xmap[y,x], self.unwarp_ymap[y,x] =  pt_imp[:2]
  
    # Compute trasnform between distorted camera image and unwarped as map
    def _compute_undist_unwarp_map(self, cam):
        print('computing undist_unwarp maps')
        self.undist_unwarp_xmap = np.zeros((self.param.h, self.param.w), np.float32)
        self.undist_unwarp_ymap = np.zeros((self.param.h, self.param.w), np.float32)
        Hinv = np.linalg.inv(self.H_unwarped)
        for y in range(self.param.h):
            for x in range(self.param.w):
                    pt_be = np.array([[x, y, 1]], dtype=np.float32).T
                    pt_imp = np.dot(cam.inv_undist_K, np.dot(Hinv, pt_be))
                    pt_imp /= pt_imp[2]
                    pt_img = cv2.projectPoints(pt_imp.T, np.zeros(3), np.zeros(3), cam.K, cam.D)[0]
                    self.undist_unwarp_xmap[y,x], self.undist_unwarp_ymap[y,x] = pt_img.squeeze()
        self.undist_unwarp_xmap_int, self.undist_unwarp_ymap_int = cv2.convertMaps(self.undist_unwarp_xmap, self.undist_unwarp_ymap, cv2.CV_16SC2)
        
    # Compute masks representing the bird eye area, for the camera and unwarped images
    def _compute_image_mask(self, cam):
        # project lfp contour to cam image
        cam_img_mask = cam.project(self.borders_isect_be_cam_lfp).squeeze()[np.newaxis].astype(np.int)
        # simplify polygon by removing uneeded vertices
        self.cam_img_mask = cv2.approxPolyDP(cam_img_mask, epsilon=1, closed=True).squeeze()[np.newaxis]
        # transform lfp contour to unwarped
        unwarped_img_mask = self.lfp_to_unwarped(cam, self.borders_isect_be_cam_lfp)[np.newaxis].astype(np.int)
        # simplify polygon by removing uneeded vertices
        self.unwarped_img_mask = cv2.approxPolyDP(unwarped_img_mask, epsilon=1, closed=True).squeeze()[np.newaxis]
        self.unwarped_img_mask2 = np.zeros((self.param.h, self.param.w), dtype=np.uint8)
        cv2.fillPoly(self.unwarped_img_mask2, [self.unwarped_img_mask], color=255)
        
        
    # Conversions between lfp and unwarped     
    def lfp_to_unwarped(self, cam, cnt_lfp):
        cnt_uv = np.array([(self.param.w/2-(_y-self.param.y0)/self.param.s, self.param.h-(_x-self.param.x0)/self.param.s) for _x, _y, _ in cnt_lfp])
        return cnt_uv

    def unwarped_to_fp(self, cam, cnt_uw):
        self.cnt_fp = np.array([((self.param.h-p[1])*self.param.s+self.param.x0, (self.param.w/2-p[0])*self.param.s+self.param.y0, 0.) for p in cnt_uw.squeeze()])
        return self.cnt_fp

    # undistort, then unwarp image
    def undist_unwarp_img(self, img, cam, use_map=True, fill_bg=None):
        if use_map:
            img_unwarped = cv2.remap(img, self.undist_unwarp_xmap_int, self.undist_unwarp_ymap_int, cv2.INTER_LINEAR)
        else:
            img_undist = cam.undistort_img(img)
            #img_unwarped = cv2.remap(img_undist, self.unwarp_xmap, self.unwarp_ymap, cv2.INTER_LINEAR) # optional
            img_unwarped = cv2.warpPerspective(img_undist, self.H_unwarped, (self.param.w, self.param.h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if fill_bg is not None:
            img_unwarped[self.unwarped_img_mask2==0] = fill_bg
        return img_unwarped

    


class UnwarpedImage:

    def draw_grid(self, img, be, cam, gridsize=0.025):

        colors = [(0,0,0), (0,0,255), (0,255,0), (255,0,0), (128, 128, 128)]
        for x in np.arange(be.param.x0, be.param.x0+be.param.dx, gridsize):
            pts_lfp = np.array([[x, -be.param.dy, 0], [x, be.param.dy, 0]])
            ps = [tuple(_p) for _p in be.lfp_to_unwarped(cam, pts_lfp).astype(int)]
            cv2.line(img, ps[0], ps[1], colors[4], 1)
        for y in np.arange(be.param.y0-be.param.dy/2, be.param.y0+be.param.dy/2, gridsize):
            pts_lfp = np.array([[be.param.x0, y, 0], [be.param.x0+be.param.dx, y, 0]])
            ps = [tuple(_p) for _p in be.lfp_to_unwarped(cam, pts_lfp).astype(int)]
            cv2.line(img, ps[0], ps[1], colors[4], 1)

        orig = [0.15, 0, 0]
        pts_lfp = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]) + orig
        pts_img = be.lfp_to_unwarped(cam, pts_lfp)
        ps = [tuple(_p) for _p in pts_img.astype(int)]
        for i in range(1,4):
            cv2.line(img, ps[0], ps[i], colors[i], 2)
        f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        cv2.putText(img, f'{pts_lfp[0,:2]}', ps[0], f, h, c, w)
        cv2.putText(img, f'{pts_lfp[1,:2]}', ps[1], f, h, c, w)
    
    def draw_path(self, img, be, cam, points):
        pts_world = np.array([[x, y, 0] for x, y in points])
        pts_img = be.lfp_to_unwarped(cam, pts_world)
        pts_img_int = [tuple(_p) for _p in pts_img.astype(int)]
        color = (0, 255, 0)
        for i in range(1,len(pts_img_int)):
            cv2.line(img, pts_img_int[i-1], pts_img_int[i], color, 2)

    def draw_trihedral(self, img, be, cam, T_trihedral_to_world, _len=0.1):
         pts_trihedral = np.array([[0, 0, 0], [_len, 0, 0], [0, _len, 0], [0, 0, _len]])
         #print(f' pts_trihedral {pts_trihedral}')
         pts_world = np.array([cv_u.transform(T_trihedral_to_world, _p) for _p in pts_trihedral])
         pts_world[:,2]=0
         #print(f' pts_world {pts_world}')
         if 0:
             pts_img = cam.project(pts_world)
             #print(f' pts_img {pts_img }')
             pts_img_undist = cam.undistort_points(pts_img.reshape(-1, 1, 2))
             pts_img_imp = np.array([np.dot(cam.inv_undist_K, [u, v, 1]) for (u, v) in pts_img_undist.squeeze()])
             pts_lfp = get_points_on_plane(pts_img_imp, cam.fp_n, cam.fp_d)
             #print(f' pts_lfp {pts_lfp }')
             #pts_img2 = be.lfp_to_unwarped(cam, pts_lfp)
         
         pts_img2 = be.lfp_to_unwarped(cam, pts_world)
         #print(f' pts_img {pts_img}')
         pts_img_int = [tuple(_p) for _p in pts_img2.astype(int)]
         #print(f' pts_img_int {pts_img_int}')
         colors = _k, _r, _g, _b = (0, 0, 0), (0,0,255), (0,255,0), (255,0,0)
         for i in range(1,4):
            cv2.line(img, pts_img_int[0], pts_img_int[i], colors[i], 2)
            #cv2.circle(img, pts_img_int[i], 5, (0,255,0), -1)
            
    def draw_cam_va(self, img, be, cam):
        #pts_lfp = be.borders_isect_be_cam_lfp
        #ps = be.lfp_to_unwarped(cam, pts_lfp).astype(int)
        #ps = ps.reshape((-1, 1, 2))
        #print(ps, img.shape)
        #cv2.polylines(img, ps, False, (0,0,255), 2)
        #cv2.drawContours(img, ps, -1, (0,255,0), 3)

        cv2.drawContours(img, be.unwarped_img_mask, -1, (0,255,255), 1)

    
