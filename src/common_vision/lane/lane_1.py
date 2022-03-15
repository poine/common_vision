import numpy as np
import cv2
#import two_d_guidance.trr.vision.utils as trr_vu
#import two_d_guidance.trr.utils as trru

import common_vision.utils as cv_u
import common_vision.bird_eye as cv_be

import pdb

# TODO: use multiple contours rather than larger one?
class Contour1Pipeline(cv_u.Pipeline):
    show_none, show_input, show_thresh, show_contour, show_be = range(5)
    def __init__(self, cam, robot_name, cache_filename=None, force_recompute=False):
        cv_u.Pipeline.__init__(self)
        #FIXME
        extr_cam_calib_path = '/home/ubuntu/work/robot_data/trilopi/camera1_extrinsics.yaml'
        cam.load_extrinsics(extr_cam_calib_path)
        
        #self.thresholder = cv_u.BinaryThresholder(thresh=180)# trr
        self.thresholder = cv_u.BinaryThresholder(thresh=190) # for roboteck
        self.contour_finder = cv_u.ContourFinder(min_area=500)
        self.floor_plane_injector = cv_u.FloorPlaneInjector()
        be_param = cv_be.NamedBirdEyeParam(robot_name)
        self.bird_eye = cv_be.BirdEye(cam,  be_param, cache_filename, force_recompute)
        self.lane_model = cv_u.LaneModel()
        self.display_mode = Contour1Pipeline.show_contour
        self.img = None
        # contour_lfp = np.array([[0.29, -0.2, 0],
        #                         [1.5, -1.4 , 0],
        #                         [1.5*np.linalg.norm([1.5, 1.4]), 0, 0],
        #                         [1.5,  1.4 , 0],
        #                         [0.29,  0.23, 0]])

        # contour_lfp = np.array([[0.208, -0.15, 0],
        #                         [1.5, -1.4 , 0],
        #                         [1.5*np.linalg.norm([1.5, 1.4]), 0, 0],
        #                         [1.5,  1.4 , 0],
        #                         [0.20,  0.165, 0]])
        # region of interest in lfp(local floor plan, front, left, up) frame
        #x0, dx, dy, dy2 = 0.14, 0.5, 0.08, 0.25
        #x0, dx, dy, dy2 = 0.105, 0.3, 0.05, 0.2
        #x0, dx, dy, dy2 = 0.105, 0.2, 0.05, 0.15
        x0, dx, dy, dy2 = 0.095, 0.2, 0.04, 0.125
        contour_lfp = np.array([[x0, -dy, 0],
                                [x0+dx, -dy2 , 0],
                                [1.2*np.linalg.norm([x0+dx, dy2]), 0, 0],
                                [x0+dx,  dy2 , 0],
                                [x0, dy, 0]])

        
        self.mask = cv_u.Mask(cam, contour_lfp)

    def set_roi(self, tl, br):
        print('roi: {} {}'.format(tl, br))
        self.tl, self.br = tl, br
        self.roi_h, self.roi_w = self.br[1]-self.tl[1], self.br[0]-self.tl[0]
        self.roi = slice(self.tl[1], self.br[1]), slice(self.tl[0], self.br[0])
        
    def _process_image(self, img, cam, stamp):
        self.img = img
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img_gray = (255-self.img_gray) # invert, for detecting black lines
        self.img_masked = cv2.bitwise_and(self.img_gray, self.img_gray, mask=self.mask.mask) 
        self.thresholder.process_gray_noflt(self.img_masked)
        self.contour_finder.process(self.thresholder.threshold)
        if self.contour_finder.cnt_max is not None:
            self.floor_plane_injector.compute(self.contour_finder.cnt_max, cam)
            self.cnt_max_blf = self.floor_plane_injector.contour_floor_plane_blf[:,:2]
            #x_min, x_max   = np.min(self.cnt_max_blf[:,0]), np.max(self.cnt_max_blf[:,0])
            #y_min, y_max = np.min(self.cnt_max_blf[:,1]), np.max(self.cnt_max_blf[:,1])
            #print('x in [{:.2f} {:.2f}] y in [{:.2f} {:.2f}]'.format(x_min, x_max, y_min, y_max))
            #print self.cnt_max_blf
            self.lane_model.fit_single_contour(self.cnt_max_blf)
            #print(self.lane_model.x_min, self.lane_model.x_max)
            self.lane_model.set_valid(True)
            self.lane_model.stamp = stamp
            #self.lane_model.set_valid(False)
        else:
            self.lane_model.set_valid(False)
        
    def draw_debug(self, cam, img_cam=None):
        return cv2.cvtColor(self.draw_debug_bgr(cam, img_cam), cv2.COLOR_BGR2RGB)

    def draw_debug_bgr(self, cam, img_cam=None):
        if self.img is None: return np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        if self.display_mode == Contour1Pipeline.show_input:
            debug_img = self.img
            #cv2.polylines(debug_img, [self.mask.contour_img] , isClosed=True, color=(0, 255, 255), thickness=2)
            self.draw_cam_scene(debug_img, cam)
        elif self.display_mode == Contour1Pipeline.show_thresh:
            #debug_img = cv2.cvtColor(self.img_gray, cv2.COLOR_GRAY2BGR)
            #debug_img = cv2.cvtColor(self.img_masked, cv2.COLOR_GRAY2BGR)
            debug_img = cv2.cvtColor(self.thresholder.threshold, cv2.COLOR_GRAY2BGR)
            cv2.polylines(debug_img, [self.mask.contour_img] , isClosed=True, color=(0, 255, 255), thickness=2)
            self.draw_cam_scene(debug_img, cam)
        elif self.display_mode == Contour1Pipeline.show_contour:
            debug_img = cv2.cvtColor(self.thresholder.threshold, cv2.COLOR_GRAY2BGR)
            self.contour_finder.draw(debug_img, draw_all=True)
            cv2.polylines(debug_img, [self.mask.contour_img] , isClosed=True, color=(0, 255, 255), thickness=2)
        elif self.display_mode == Contour1Pipeline.show_be:
            debug_img = self.draw_be_scene(cam)

        f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 2
        cv2.putText(debug_img, 'Lane #1', (20, 40), f, h, c, w)
        self.draw_timing(debug_img)
        
        if self.lane_model.is_valid() and self.display_mode not in [Contour1Pipeline.show_be]:
            x_min, x_max = self.lane_model.x_min, self.lane_model.x_max
            #x_min, x_max = 0.15, 12.
            self.lane_model.draw_on_cam_img(debug_img, cam, l0=x_min, l1=x_max, color=(128, 128, 0))
        return debug_img


    def draw_cam_scene(self, img, cam):
        orig = [0.15, 0, 0]
        pts_lfp = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]) + orig
        pts_img = cam.project(pts_lfp)
        ps = [tuple(pts_img[_i,0].astype(int)) for _i in range(len(pts_img))]
        cv2.line(img, ps[0], ps[1], (0,0,255), 2)
        cv2.line(img, ps[0], ps[2], (0,255,0), 2)
        cv2.line(img, ps[0], ps[3], (255,0,0), 2)
        cv2.polylines(img, [self.mask.contour_img] , isClosed=True, color=(0, 255, 255), thickness=2)
        f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2
        cv2.putText(img, f'{orig}', ps[0], f, h, c, w)
    
    def draw_be_scene(self, cam):
        return np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        be_img = np.zeros((self.bird_eye.h, self.bird_eye.w, 3), dtype=np.uint8)
        
        # draw blf frame axis
        pts_blf = np.array([[0, 0, 0], [1, 0, 0],
                            [0, 0, 0], [0, 1, 0]], dtype=np.float32)
        pts_img = self.bird_eye.lfp_to_unwarped(cam, pts_blf)
        color = (128, 0, 0)
        for i in range(len(pts_img)-1):
            cv2.line(be_img, tuple(pts_img[i].squeeze().astype(int)), tuple(pts_img[i+1].squeeze().astype(int)), color, 4)

        if self.lane_model.is_valid():
            self.bird_eye.draw_lane(cam, be_img, self.lane_model, self.lane_model.x_min, self.lane_model.x_max)
        

            
        #debug_img = self.bird_eye.draw_debug(cam, img=debug_img, lane_model=self.lane_model)
        debug_img = cv_u.change_canvas(be_img, cam.h, cam.w, border_color=(128, 128, 128))
        
        return debug_img
