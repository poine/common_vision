import numpy as np
#import matplotlib.pyplot as plt, matplotlib
import matplotlib, matplotlib.pyplot as plt, mpl_toolkits.mplot3d

import cv2
import pdb
import common_vision.utils as cv_u


def plot_images(imgs, imgs_path, img_points, rets, rep_img_pts, nc=4, cb_geom=(8,6)):
    nr = int(np.ceil(len(imgs)/nc))+1
    for i, img in enumerate(imgs):
        ax = plt.gcf().add_subplot(nr, nc, i+1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.drawChessboardCorners(img_rgb, cb_geom, img_points[i], rets[i])
        #for p1, p2  in zip(img_points[i], rep_img_pts[i]):
            #ax.add_patch(matplotlib.patches.Circle(p1,10, color='b'))
            #ax.add_patch(matplotlib.patches.Circle(p2,10, color='r', alpha=0.5))
        #plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.imshow(img_rgb)
        #plt.title(imgs_path[i])
        ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)

def plot_images2(imgs, imgs_path, img_points, rets, rep_img_pts, cb_geom=(8,6)):
    for i, (img, _path) in enumerate(zip(imgs, imgs_path)):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #cv2.drawChessboardCorners(img_bgr, cb_geom, img_points[i], rets[i])
        radius, color1, color2, thickness = 1, (0, 255, 0), (0, 0, 255), 1
        for p1, p2  in zip(img_points[i], rep_img_pts[i]):
            #pdb.set_trace()
            cv2.circle(img_bgr, tuple(p1.squeeze()),  radius, color1, thickness)
            cv2.circle(img_bgr, tuple(p2.squeeze()),  radius, color2, thickness)

        cv2.imshow('{}'.format(_path), img_bgr)
        cv2.waitKey(0)

def plot_rep_err(imgs, imgs_path, img_points, rets, rep_img_pts):
    pass




# 3D scene
def draw_thriedra(ax, T_thriedra_to_world__thriedra, alpha=1., colors=['r', 'g', 'b'], scale=1., ls='-', id=None):
    ''' Draw thriedra in w frame '''
    t, R = cv_u.tR_of_T(T_thriedra_to_world__thriedra)
    for i in range(3):
        p1 = t + scale*R[:,i] # aka p1 = t + np.dot(R, v) with v axis vector ([1 0 0], [0 1 0], [0 0 1])
        ax.plot([t[0], p1[0]], [t[1], p1[1]], [t[2], p1[2]], ls, color=colors[i], alpha=alpha)
    if id is not None:
        annotate3D(ax, s=str(id), xyz= T_thriedra_to_world__thriedra[:3,3], fontsize=10, xytext=(-3,3),
                   textcoords='offset points', ha='right',va='bottom')

def draw_points(ax, pts_coord, pts_id):
    ax.scatter(pts_coord[:,0], pts_coord[:,1], pts_coord[:,2], marker='o')
    for coor, _id in zip(pts_coord, pts_id):
        annotate3D(ax, s=str(_id), xyz=coor, fontsize=10, xytext=(-3,3),
                   textcoords='offset points', ha='right',va='bottom')

def draw_camera(ax, T_c_to_w__c, id=None, color='k'):
    ''' draw a camera as a pyramid '''
    draw_thriedra(ax, T_c_to_w__c, scale=0.1, id=id)
    w, h, d = 0.1, 0.05, 0.25
    pts_c = [[ 0,  0, 0, 1],
             [ w,  h, d, 1],
             [-w,  h, d, 1],
             [ 0,  0, 0, 1],
             [ w, -h, d, 1],
             [-w, -h, d, 1],
             [ 0,  0, 0, 1],
             [ w,  h, d, 1],
             [ w, -h, d, 1],
             [-w, -h, d, 1],
             [-w,  h, d, 1]]
    pts_w = np.array([np.dot(T_c_to_w__c, pt_c) for pt_c in pts_c])
    ax.plot(pts_w[:,0], pts_w[:,1], pts_w[:,2], color=color)

def set_3D_axes_equal(ax=None):
    '''
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    if ax is None: ax = plt.gca()

    x_limits, y_limits, z_limits = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# http://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot
class Annotation3D(matplotlib.text.Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        matplotlib.text.Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = mpl_toolkits.mplot3d.proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        matplotlib.text.Annotation.draw(self, renderer)


def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)
