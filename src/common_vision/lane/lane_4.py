# -*- coding: utf-8 -*-

import numpy as np
import cv2
import common_vision.utils as cv_u
# FIXME: change that to our own utils when all is available
import two_d_guidance.trr.vision.utils as trr_vu
import two_d_guidance.trr.utils as trr_u
import pdb

class Pipeline(trr_vu.Pipeline):
    show_none, show_input, show_lines, show_be, show_summary = range(5)
    def __init__(self, cam, robot_name):
        trr_vu.Pipeline.__init__(self)
        be_param = trr_vu.NamedBirdEyeParam(robot_name)
        self.cam = cam
        self.set_roi((0, 0), (cam.w, cam.h))  
        self.bird_eye = trr_vu.BirdEyeTransformer(cam, be_param)
        self.line_finder = trr_vu.HoughLinesFinder(self.bird_eye.mask_unwraped)
        self.lane_model = trr_u.LaneModel()
        
    def set_roi(self, tl, br): pass

    def _make_points_of_lines(self):
        pts_be = []
        for _l in self.line_finder.lines.squeeze(axis=1):
            p1, p2 = _l[:2], _l[2:]
            if 0:  # just both end of the line
                pts_be.append(p1); pts_be.append(p2)
            else:  # add all segment points
                n_pt = np.linalg.norm(p2-p1)/1. # resolution, one point per pixel
                ps_line = np.stack([np.linspace(p1[i], p2[i], n_pt) for i in range(2)], axis=-1)
                for p in ps_line: pts_be.append(p)
        return np.array(pts_be)
    
    def fit_lines(self):
        pts_be = self._make_points_of_lines()
        pts_lfp = self.bird_eye.unwarped_to_fp(None, pts_be)
        xs, ys = pts_lfp[:,0], pts_lfp[:,1]
        order = 3
        self.lane_model.coefs, self._res, rank, _singular, _rcond = np.polyfit(xs, ys, order, full=True);#, w=weights)
        print self._res
        self.lane_model.x_min, self.lane_model.x_max = np.min(xs), np.max(xs)
        self.lane_model.set_valid(True)

    def fit_lines_ransac(self, order=3, n=20, k=20, t=10., d=100, f=0.8):
        # Thanks https://en.wikipedia.org/wiki/Random_sample_consensus
        # n – minimum number of data points required to fit the model
        # k – maximum number of iterations allowed in the algorithm
        # t – threshold value to determine when a data point fits a model
        # d – number of close data points required to assert that a model fits well to data
        # f – fraction of close data points required
        pts_be = self._make_points_of_lines()
        pts_lfp = self.bird_eye.unwarped_to_fp(None, pts_be)
        xs, ys = pts_lfp[:,0], pts_lfp[:,1]
        
        besterr = np.inf
        bestfit = None
        for kk in xrange(k):
            maybeinliers = np.random.randint(len(xs), size=n)
            maybemodel = np.polyfit(xs[maybeinliers], ys[maybeinliers], order)
            alsoinliers = np.abs(np.polyval(maybemodel, xs)-ys) < t
            if sum(alsoinliers) > d and sum(alsoinliers) > len(xs)*f:
                bettermodel = np.polyfit(xs[alsoinliers], ys[alsoinliers], order)
                thiserr = np.sum(np.abs(np.polyval(bettermodel, xs[alsoinliers])-ys[alsoinliers]))
                if thiserr < besterr:
                    bestfit = bettermodel
                    besterr = thiserr

        if bestfit is None:
            self.lane_model.set_invalid()
        else:
            self.lane_model.x_min, self.lane_model.x_max = np.min(xs), np.max(xs)
            self.lane_model.coefs = bestfit
            self.lane_model.set_valid(True)
            self._res = besterr
        #pdb.set_trace()
        return bestfit
 
    def _process_image(self, img, cam, stamp):
        self.img = img
        self.img_unwarped = self.bird_eye.undist_unwarp_map(img, cam)
        self.line_finder.process(self.img_unwarped)
        # fit lines
        if self.line_finder.lines is not None and len(self.line_finder.lines) > 0:
            self.fit_lines()
            #self.fit_lines_ransac()
        else: self.lane_model.set_invalid()#self.lane_model.set_valid(False)
            
    def draw_debug(self, cam, img_cam=None):
        return cv2.cvtColor(self.draw_debug_bgr(cam, img_cam), cv2.COLOR_BGR2RGB)

    def draw_debug_bgr(self, cam, img_cam=None, border_color=128):
        if self.img is None: return np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        if self.display_mode == Pipeline.show_input:
            debug_img = self.img
        elif self.display_mode == Pipeline.show_lines:
            debug_img = self._draw_lines(cam)
        elif self.display_mode == Pipeline.show_be:
            debug_img = self._draw_be(cam)
        elif self.display_mode == Pipeline.show_summary:
            debug_img = self._draw_summary(cam)
            
        self._draw_HUD(debug_img)
        return debug_img # we return a BGR image
    
    def _draw_HUD(self, debug_img):
        f, h, c, w = cv_u.get_default_cv_text_params()
        h1, c1, dy = 1., (220, 130, 120), 30
        cv2.putText(debug_img, 'Lane#4', (20, 40), f, h, c, w)
        self.draw_timing(debug_img, x0=360, y0=40)

    def _draw_summary(self, cam):
        h1, w1 = self.img.shape[:2]
        #h2, w2 = self.img_unwarped.shape[:2]
        h2, w2 = self.bird_eye.h, self.bird_eye.w 
        #print('cam {} {} be {} {}'.format(w1, h1, w2, h2))
        w, h = w1+h2, max(h1, 2*w2)
        debug_img = np.zeros((h, w, 3), dtype=np.uint8)
        debug_img[:h1,:w1] = self.img
        #debug_img[:h2,w1:] = cv2.rotate(self.img_unwarped, rotateCode=1)
        debug_img[:w2,w1:] = np.flipud(np.fliplr(np.rot90(self.img_unwarped)))
        #debug_img[w2:2*w2,w1:] = np.flipud(np.fliplr(np.rot90(cv2.cvtColor(self.line_finder.edges, cv2.COLOR_GRAY2BGR))))
        img_edges = cv2.cvtColor(self.line_finder.edges, cv2.COLOR_GRAY2BGR)
        #cv2.fillPoly(img_edges, pts=self.line_finder.mask, color=(0, 255, 0))
        self.line_finder.draw(img_edges)
        #print self.lane_model.coefs
        if self.lane_model.is_valid() and self.lane_model.coefs is not None: self.bird_eye.draw_lane(cam, img_edges, self.lane_model)
        #pdb.set_trace()
        debug_img[w2:2*w2,w1:] = np.flipud(np.fliplr(np.rot90(img_edges)))
        return debug_img

    def _draw_lines(self, cam):
        debug_img = cv2.cvtColor(self.line_finder.edges, cv2.COLOR_GRAY2BGR)
        #self.line_finder.draw(debug_img)
        # try:
        #     debug_img = self.bird_eye.draw_debug(cam, self.img_unwarped, None, None)#self.lane_model, self.cnts_be)
        # except AttributeError:
        #     debug_img = np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        return debug_img

    def _draw_be(self, cam):
        try:
            debug_img = self.bird_eye.draw_debug(cam, self.img_unwarped, None, None)#self.lane_model, self.cnts_be)
        except AttributeError:
            debug_img = np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        return debug_img
