import numpy as np
import matplotlib.pyplot as plt, matplotlib
import cv2
import pdb

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
