import math, numpy as np, cv2
try:
    import tf.transformations
except ImportError:
    print('common_vision.utils.py: can not find tf')
    
import os, glob, tarfile, yaml
import time

import pdb

import logging
LOG = logging.getLogger('common_vision.utils')

# Read an array of points - this is used for extrinsic calibration
def read_point(yaml_data_path):
    with open(yaml_data_path, 'r') as stream:
        ref_data = yaml.load(stream)
    pts_name, pts_img, pts_world = [], [], []
    for _name, _coords in ref_data.items():
        pts_img.append(_coords['img'])
        pts_world.append(_coords['world'])
        pts_name.append(_name)
    return pts_name, np.array(pts_img, dtype=np.float64), np.array(pts_world, dtype=np.float64)



'''
Retrieve all images (as grayscale) in the given directory
'''
def load_images_in_dir(_dir, _prefix, read_as=cv2.IMREAD_GRAYSCALE, idxs=None):
    img_glob = '{}/{}*'.format(_dir, _prefix)
    LOG.info(" loading images: {}".format(img_glob))
    img_path = glob.glob(img_glob)
    img_path.sort()
    if idxs is not None:
        new_path = []
        for path in img_path:
            base = os.path.basename(path)
            noext = os.path.splitext(base)[0]
            if int(noext.split('_')[-1])  in idxs:
                new_path.append(path)
                #pdb.set_trace()
                #print(noext)
        img_path = new_path
    LOG.info(" found {} images".format(len(img_path)))
    images = [cv2.imread(p, read_as) for p in img_path]
    return images, img_path

'''
Retrieve all images (as grayscale) in the given tarfile
'''
def load_images_in_tarfile(tar_filename, img_prefix, read_as=cv2.IMREAD_GRAYSCALE):
    archive = tarfile.open(tar_filename, 'r')
    imgs, img_filenames = [], []
    for f in archive.getnames():
        if f.startswith(img_prefix):
            filedata = archive.extractfile(f).read()
            file_bytes = np.asarray(bytearray(filedata), dtype=np.uint8)
            imgs.append(cv2.imdecode(file_bytes, read_as))
            img_filenames.append(f)
    LOG.info(" found {} images".format(len(imgs)))
    return imgs, img_filenames




'''
   Compute reprojection error using opencv projection
'''
def compute_reprojection_error(img_points, object_points, cmtx, distk, rvecs, tvecs):
    rep_pts, rep_errs = [], []
    for img_pts, obj_pts, rvec, tvec in zip(img_points, object_points, rvecs, tvecs):
        img_points2, jac = cv2.projectPoints(obj_pts, rvec, tvec, cmtx, distk)
        rep_pts.append(img_points2)
        rep_errs.append(np.linalg.norm(img_points2-img_pts, axis=2))
    _sses = np.concatenate(rep_errs)
    _rep_err_rms = np.mean(np.sqrt(np.mean(_sses**2)))
    LOG.info(" reprojection error")
    LOG.info("   min, max, rms: {:.3f} {:.3f} {:.3f} pixels".format(np.min(_sses), np.max(_sses), _rep_err_rms))
    return rep_pts, rep_errs

# for printing on images
def get_default_cv_text_params(): return cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 2


# Stolen from smocap


def norm_angle(_a):
    while _a <= -math.pi: _a += 2*math.pi
    while _a >   math.pi: _a -= 2*math.pi    
    return _a


# fucked up python 2 vs 3
import numpy
# epsilon for testing whether a number is close to zero
_EPS = numpy.finfo(float).eps * 4.0
def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = numpy.array(quaternion[:4], dtype=numpy.float64, copy=True)
    nq = numpy.dot(q, q)
    if nq < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    return numpy.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=numpy.float64)

def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.

    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True

    """
    q = numpy.empty((4, ), dtype=numpy.float64)
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    t = numpy.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q





# Transforms
def T_of_t_rpy(t, rpy):
    T = tf.transformations.euler_matrix(rpy[0], rpy[1], rpy[2], 'sxyz')
    T[:3,3] = t
    return T

def t_rpy_of_T(T):
    t = T[:3,3]
    rpy = tf.transformations.euler_from_matrix(T, 'sxyz')
    return t, rpy

def T_of_t_q(t, q):
    #T = tf.transformations.quaternion_matrix(q)
    T = quaternion_matrix(q)
    T[:3,3] = t
    return T

def T_of_t_R(t, R):
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
    return T

def T_of_t_r(t, r):
    R, _ = cv2.Rodrigues(r)
    return T_of_t_R(t, R)

def tq_of_T(T):
    #return T[:3, 3], tf.transformations.quaternion_from_matrix(T)
    return T[:3, 3], quaternion_from_matrix(T)

def tR_of_T(T):
    return T[:3,3], T[:3,:3]

def tr_of_T(T):
    ''' return translation and rodrigues angles from a 4x4 transform matrix '''
    r, _ = cv2.Rodrigues(T[:3,:3])
    return T[:3,3], r.squeeze()


def transform(a_to_b_T, p_a):
    return np.dot(a_to_b_T[:3,:3], p_a) + a_to_b_T[:3,3]

# TF messages
def list_of_position(p): return (p.x, p.y, p.z)
def list_of_orientation(q): return (q.x, q.y, q.z, q.w)
def t_q_of_transf_msg(transf_msg):
    return list_of_position(transf_msg.translation), list_of_orientation(transf_msg.rotation)
def _position_and_orientation_from_T(p, q, T):
    p.x, p.y, p.z = T[:3, 3]
    q.x, q.y, q.z, q.w = tf.transformations.quaternion_from_matrix(T)
def _T_from_landmark_transform(lmt): # cartographer_ros_msgs.msg.LandmarkList
    return  T_of_t_q(list_of_position(lmt.position), list_of_orientation(lmt.orientation))
    




class Mask:
    def __init__(self, cam, blf_contour=None):
        self.mask = np.zeros((cam.h, cam.w), np.uint8)
        if blf_contour is not None:
            self.load_blf_contour(cam, blf_contour)
        
    def load_blf_contour(self, cam, blf_contour):
        self.contour_img = cam.project(blf_contour).astype(np.int64).squeeze()
        cv2.fillPoly(self.mask, [self.contour_img], color=255)

    



        
# Timing of image processing
class Pipeline:
    def __init__(self):
        self.skipped_frames = 0
        self.last_seq = None
        self.last_stamp = None
        self.cur_fps = 0.
        self.min_fps, self.max_fps, self.lp_fps = np.inf, 0, 0.1
        self.last_processing_duration = None
        self.min_proc, self.max_proc, self.lp_proc = np.inf, 0, 1e-6
        self.idle_t = 0.
        self.k_lp = 0.9 # low pass coefficient
        
    def process_image(self, img, cam, stamp, seq):
        if self.last_stamp is not None:
            _dt = (stamp - self.last_stamp).to_sec()
            if np.abs(_dt) > 1e-9:
                self.cur_fps = 1./_dt
                self.min_fps = np.min([self.min_fps, self.cur_fps])
                self.max_fps = np.max([self.max_fps, self.cur_fps])
                self.lp_fps  = self.k_lp*self.lp_fps+(1-self.k_lp)*self.cur_fps
        self.last_stamp = stamp
        if self.last_seq is not None:
            self.skipped_frames += seq-self.last_seq-1
        self.last_seq = seq

        _start = time.time()
        self._process_image(img, cam, stamp)
        _end = time.time()

        self.last_processing_duration = _end-_start
        self.min_proc = np.min([self.min_proc, self.last_processing_duration])
        self.max_proc = np.max([self.max_proc, self.last_processing_duration])
        self.lp_proc = self.k_lp*self.lp_proc+(1-self.k_lp)*self.last_processing_duration
        self.idle_t = 1./self.lp_fps - self.lp_proc

    def draw_timing(self, img, x0=280, y0=20, dy=35, h=0.75, color_bgr=(220, 220, 50)):
        f, c, w = cv2.FONT_HERSHEY_SIMPLEX, color_bgr, 2
        try: 
            txt = 'fps: {:.1f} (min {:.1f} max {:.1f})'.format(self.lp_fps, self.min_fps, self.max_fps)
            cv2.putText(img, txt, (x0, y0), f, h, c, w)
            txt = 'skipped: {:d} (cpu {:.3f}/{:.3f}s)'.format(self.skipped_frames, self.lp_proc, 1./self.lp_fps)
            cv2.putText(img, txt, (x0, y0+dy), f, h, c, w)
        except AttributeError: pass

    def draw_debug(self, cam, img_cam=None):
        return cv2.cvtColor(self.draw_debug_bgr(cam, img_cam), cv2.COLOR_BGR2RGB)
        
#
# Images
#
       
class ImgPublisher:
    def __init__(self, cam, img_topic = "/trr_vision/start_finish/image_debug"):
        rospy.loginfo(' publishing image on ({})'.format(img_topic))
        self.image_pub = rospy.Publisher(img_topic, sensor_msgs.msg.Image, queue_size=1)
        self.bridge = cv_bridge.CvBridge()
        
    def publish(self, producer, cam, encoding="rgb8"):
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(producer.draw_debug(cam), encoding))

class CompressedImgPublisher:
    def __init__(self, cam, img_topic):
        img_topic = img_topic + "/compressed"
        rospy.loginfo(' publishing image on ({})'.format(img_topic))
        self.image_pub = rospy.Publisher(img_topic, sensor_msgs.msg.CompressedImage, queue_size=1)
        
    def publish(self, model, data):
        img_rgb = model.draw_debug(data)
        self.publish2(img_rgb)
        
    def publish2(self, img_rgb):
        img_bgr =  cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        msg = sensor_msgs.msg.CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', img_bgr)[1]).tostring()
        self.image_pub.publish(msg)
        



# display a downscaled version of an image
def imshow_scaled(img, txt="", max_size=1500):
    scale = max_size/max(img.shape)
    if scale < 1:
        img2 = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
    else:
        img2 = img
    cv2.imshow(txt, img2)

# display images as mosaic
def imshow_mosaic(imgs, txt, size=640, ncol=2):
    nrow = int(np.ceil(len(imgs)/ncol))
    iw, ih = imgs[0].shape[:2]
    scale = size/max([iw, ih])
    miw, mih = int(iw*scale), int(ih*scale) 
    blank_image = np.zeros(shape=[ncol*miw, nrow*mih, 3], dtype=np.uint8)
    for i in range(len(imgs)):
        _im =  cv2.resize(imgs[i], (mih, miw))
        ic, ir = i%ncol, int(i/ncol)
        blank_image[ir*miw:(ir+1)*miw, ic*mih:(ic+1)*mih] = _im
    
    cv2.imshow(txt, blank_image)
