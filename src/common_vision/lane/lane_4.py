
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

        
    def _process_image(self, img, cam, stamp):
        self.img = img
        self.img_unwarped = self.bird_eye.undist_unwarp_map(img, cam)
        self.line_finder.process(self.img_unwarped)

        
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
        print('cam {} {} be {} {}'.format(w1, h1, w2, h2))
        w, h = w1+h2, max(h1, 2*w2)
        debug_img = np.zeros((h, w, 3), dtype=np.uint8)
        debug_img[:h1,:w1] = self.img
        #debug_img[:h2,w1:] = cv2.rotate(self.img_unwarped, rotateCode=1)
        debug_img[:w2,w1:] = np.flipud(np.fliplr(np.rot90(self.img_unwarped)))
        #debug_img[w2:2*w2,w1:] = np.flipud(np.fliplr(np.rot90(cv2.cvtColor(self.line_finder.edges, cv2.COLOR_GRAY2BGR))))
        img_edges = cv2.cvtColor(self.line_finder.edges, cv2.COLOR_GRAY2BGR)
        cv2.fillPoly(img_edges, pts=self.line_finder.mask, color=(0, 255, 0))
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
