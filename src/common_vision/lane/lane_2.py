import numpy as np
import cv2
import shapely

import common_vision.utils as cv_u
import common_vision.bird_eye as cv_be

class Lane2Pipeline(cv_u.Pipeline):
    show_none, show_input, show_mask, show_contour, show_be = range(5)
    def __init__(self, cam, robot_name, cache_filename=None, force_recompute=False):
        cv_u.Pipeline.__init__(self)
        #FIXME
        extr_cam_calib_path = '/home/ubuntu/work/robot_data/trilopi/camera1_extrinsics.yaml'
        cam.load_extrinsics(extr_cam_calib_path)

        self.bird_eye = cv_be.BirdEye(cam, cv_be.BeParamTrilopi(), cache_filename='/tmp/be_cfg.npz', force_recompute=False)

        self.thresholder = cv_u.BinaryThresholder(thresh=190) # for roboteck
        self.contour_finder = cv_u.ContourFinder(min_area=500)
        
        self.lane_model = cv_u.LaneModel()
        self.display_mode = Lane2Pipeline.show_contour
        self.img = None

    def _process_image(self, img, cam, stamp):
        self.img = img
        self.img_unwarped = self.bird_eye.undist_unwarp_img(self.img, cam)

        #bgrdash = self.img_unwarped.astype(np.float)/255.
        #self.K_chan = 1 - np.max(bgrdash, axis=2)
        #self.K_chan2 = (self.K_chan*255.).astype(np.uint8)
        #print(bgrdash.shape, self.K_chan.shape)
        self.img_unwarped_hsv = cv2.cvtColor(self.img_unwarped, cv2.COLOR_BGR2HSV)
        self.img_unwarped_v = self.img_unwarped_hsv[:,:,2]
        
        self.img_gray = self.img_unwarped_v#cv2.cvtColor(self.img_unwarped, cv2.COLOR_BGR2GRAY)
        self.img_gray = (255-self.img_gray) # invert, for detecting black lines

        # FIXME, make that in bird eye
        self.mask = np.zeros_like(self.img_gray)
        cv2.fillPoly(self.mask, [self.bird_eye.unwarped_img_mask], color=255)

        self.img_masked = cv2.bitwise_and(self.img_gray, self.img_gray, mask=self.mask)
        self.thresholder.process_gray_noflt(self.img_masked)
        self.contour_finder.process(self.thresholder.threshold)

        if self.contour_finder.cnt_max is not None:
            self.cnt_lfp = self.bird_eye.unwarped_to_fp(cam, self.contour_finder.cnt_max)
            #print(cnt_lfp)
            self.lane_model.fit_single_contour(self.cnt_lfp)
            #print(self.contour_finder.cnt_max.reshape((-1, 2)))
            if 1:
                poly_lfp = shapely.geometry.Polygon(self.cnt_lfp[:,:2])
                poly_lfp_eroded = poly_lfp#poly_lfp.buffer(-0.005)
                self.cnt_eroded_lfp = np.array([poly_lfp_eroded.exterior.coords.xy]).T
                tmp = np.zeros((len(self.cnt_eroded_lfp), 3)); tmp[:,:2] = self.cnt_eroded_lfp.squeeze()
                self.cnt_img_3 = np.array([self.bird_eye.lfp_to_unwarped(cam, tmp).astype(int)])
                #print(self.cnt_lfp.shape, tmp.shape)
                self.lane_model.fit_single_contour(tmp)
            if 0:
                poly_cnt_img = shapely.geometry.Polygon(self.contour_finder.cnt_max.reshape((-1, 2)))
                poly_cnt_img_eroded = poly_cnt_img.buffer(-15.)
                #print(poly_cnt_img.is_valid, poly_cnt_img.area, new_poly.area)
                self.cnt_img_2 = np.array([poly_cnt_img_eroded.exterior.coords.xy]).T.astype(int) #np.array(new_poly).T.astype(int)
                #print(f'img2 {self.cnt_img_2.shape} img3{self.cnt_eroded_lfp.shape}')
                #print(self.cnt_lfp.shape, self.cnt_img_2.shape)
                #self.lane_model.fit_single_contour(self.cnt_img_2[0])
                #print(self.lane_model.x_min, self.lane_model.x_max)
            self.lane_model.stamp = stamp
            self.lane_model.set_valid(True)
        else:
            self.lane_model.set_valid(False)

    def draw_debug_bgr(self, cam, img_cam=None):
        if self.img is None: return np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        if self.display_mode == Lane2Pipeline.show_input:
            debug_img = self.img
        elif self.display_mode == Lane2Pipeline.show_mask:
            #debug_img = cv2.cvtColor(self.img_masked, cv2.COLOR_GRAY2BGR)
            debug_img = cv2.cvtColor(self.img_unwarped_hsv[:,:,2], cv2.COLOR_GRAY2BGR)#self.K_chan2
        elif self.display_mode == Lane2Pipeline.show_contour:
            debug_img = cv2.cvtColor(self.thresholder.threshold, cv2.COLOR_GRAY2BGR)
            if self.contour_finder.cnt_max is not None:
                cv2.drawContours(debug_img, self.bird_eye.unwarped_img_mask, -1, (0,255,255), 2)
                cv2.drawContours(debug_img, self.contour_finder.cnt_max, -1, (0,0,255), 4)
                #try:
                #cv2.drawContours(debug_img, [self.cnt_img_2], -1, (0,255,0), 2)
                #except:
                #print('failed to draw new poly')
                #import pdb; pdb.set_trace()
                #print(self.cnt_img_3, self.contour_finder.cnt_max.shape)
        else:
            debug_img = cv2.bitwise_and(self.img_unwarped, self.img_unwarped, mask=self.mask)
            green = (0, 177, 64)

        if self.lane_model.is_valid() and self.display_mode not in [Lane2Pipeline.show_input]:
            xs = np.linspace(self.lane_model.x_min, self.lane_model.x_max, 20); ys = self.lane_model.get_y(xs)
            pts_lfp = np.array([[x, y, 0] for x, y in zip(xs, ys)])
            pts_img = self.bird_eye.lfp_to_unwarped(cam, pts_lfp)
            ps = [tuple(_p) for _p in pts_img.astype(int)]
            for i in range(len(ps)-1):
                cv2.line(debug_img, ps[i], ps[i+1], (255,0,0), 3)
            
        f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 2
        cv2.putText(debug_img, 'Lane #2', (20, 40), f, h, c, w)
        self.draw_timing(debug_img)
        return debug_img
