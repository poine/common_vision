import math, numpy as np, cv2, tf.transformations
import glob, tarfile, yaml

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
def load_images_in_dir(_dir, _prefix, read_as=cv2.IMREAD_GRAYSCALE):
    img_glob = '{}/{}*'.format(_dir, _prefix)
    LOG.info(" loading images: {}".format(img_glob))
    img_path = glob.glob(img_glob)
    img_path.sort()
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
    T = tf.transformations.quaternion_matrix(q)
    T[:3,3] = t
    return T

def T_of_t_R(t, R):
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
    return T

def T_of_t_r(t, r):
    R, _ = cv2.Rodrigues(r)
    return T_of_t_R(t, R)

def tq_of_T(T):
    return T[:3, 3], tf.transformations.quaternion_from_matrix(T)

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




class Mask:
    def __init__(self, cam, blf_contour=None):
        self.mask = np.zeros((cam.h, cam.w), np.uint8)
        if blf_contour is not None:
            self.load_blf_contour(cam, blf_contour)
        
    def load_blf_contour(self, cam, blf_contour):
        self.contour_img = cam.project(blf_contour).astype(np.int64).squeeze()
        cv2.fillPoly(self.mask, [self.contour_img], color=255)

    
