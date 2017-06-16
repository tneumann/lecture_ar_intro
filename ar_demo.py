import numpy as np
import cv2
import pyglet
from pyglet.gl import *

from importers import Wavefront
from util import numpy_array_to_gl_matrix, numpy_to_pyglet_image


def locate_marker(img, marker, threshold=0.75, 
                  border=4, min_num_points=20, approx_eps=5, 
                  threshold_blocksize=51, threshold_C=10):
    # convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # threshold to obtain binary segmentation
    img_bin = cv2.adaptiveThreshold(
        img_gray, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 
        threshold_blocksize, threshold_C
    )
    # find connected contours in binary image
    contours, hierarchy = cv2.findContours(
        img_bin.copy(), # need to pass a copy of the image, 
                        # because the image is modified by the function
        cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # filter contours with too few points
    contours_filtered = [c for c in contours if len(c) > min_num_points]
    # approximate contours
    contours_approx = [
        cv2.approxPolyDP(c, approx_eps, True) for c in contours_filtered]
    # only accept quads (contours with 4 points)
    quads = [c for c in contours_approx if len(c) == 4]

    # setup quad points on the reference plane (Z=0)
    marker_width, marker_height = marker.shape[1], marker.shape[0]
    points_plane = np.array([
        (0, 0),
        (0, marker_height),
        (marker_width, marker_height),
        (marker_width, 0),
        ], np.float)
    # only consider central region in marker - crop marker image
    marker_cropped = marker[border:-border, border:-border]

    for quad in quads:
        # find homography that warps detected points into the marker image
        H, _ = cv2.findHomography(
            quad.reshape(-1, 2).astype(np.float),
            points_plane)
        # perform this warp
        # need to invert binary image since we use THRESH_BINARY_INV above
        unwarped = 255 - cv2.warpPerspective(
            img_bin, H, (marker_width, marker_height),
            flags=cv2.INTER_NEAREST)
        # remove border in image
        unwarped = unwarped[border:-border, border:-border]
        for rot in xrange(4):
            rotated = np.rot90(unwarped, rot)
            overlap = (rotated == marker_cropped).mean()
            if overlap > threshold:
                quad = quad.reshape(4, 2)
                quad = np.roll(quad, rot, axis=0)
                return quad.reshape(-1, 2), rotated

    return None, None


def update(dt=None):
    # sorry ... using global variables for simplicity here...
    global quad, img, R, tvec, K, unwarped_marker
    # get image from the camera
    status, img = webcam.read()
    # make it half size
    img = cv2.resize(img, None, None, 1/2., 1/2.)
    # opencv reads camera images in blue, green, red channel order,
    # we want RGB - red, green, blue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # hardcoded intrinsic parameters of the camera
    height, width, num_channels = img.shape
    fx = fy = 504
    cx =  width / 2.
    cy = height / 2.

    # try to locate the marker
    quad, unwarped_marker = locate_marker(img, marker)

    if quad is not None:
        # intrinsic camera matrix
        K = np.array([
            (fx,  0,  cx),
            (0,  fy,  cy),
            (0,   0,   1),
        ], np.float)
        status, rvec, tvec = cv2.solvePnP(
            points_plane3d, 
            quad.reshape(-1, 2).astype(np.float), 
            K, None)
        R, _ = cv2.Rodrigues(rvec)



# load marker
marker = cv2.cvtColor(cv2.imread("marker.png"), cv2.COLOR_BGR2GRAY)
marker = cv2.resize(marker, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
# marker points in 3D
points_plane3d = np.array([
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0) ], np.float)

# initialize video capture
webcam = cv2.VideoCapture(0)
# load 3D object to show
monkey = Wavefront("monkey.obj")

update() # call update once so that global variables are initialized

# make a window of same size as the camera image
window = pyglet.window.Window(
    width=img.shape[1],
    height=img.shape[0]
)
# schedule update at 30fps
pyglet.clock.schedule_interval(update, 1/30.0)


@window.event
def on_draw():
    window.clear()
    # setup orthogonal projection to draw image
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, img.shape[1], 0, img.shape[0], -1, 1)
    # draw image
    glColor3f(1, 1, 1)
    numpy_to_pyglet_image(img).blit(0, 0)

    # marker detected?
    if quad is not None:
        numpy_to_pyglet_image(unwarped_marker).blit(0, 0)
        # draw detected quad in pixel space
        glColor3f(0, 1, 0)
        quad_points = quad.reshape(-1, 2)
        quad_points[:, 1] = img.shape[0] - quad_points[:, 1] # invert y axis
        glColor3f(0, 1, 0)
        pyglet.graphics.draw(
            len(quad), GL_LINE_LOOP,
            ('v2f', quad.ravel()),
        )

        # build OpenGL projection matrix
        near = 0.01
        far = 100.
        P = np.zeros((4, 4))
        P[0, 0] = K[0, 0] / K[0, 2]
        P[1, 1] = -K[1, 1] / K[1, 2] # invert y axis
        P[2, 2] = (far + near) / (far - near)
        P[2, 3] = -2 * far * near / (far - near) # invert z axis
        P[3, 2] = 1.

        # setup 3D view
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixd(numpy_array_to_gl_matrix(P))

        # setup lights
        glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat * 4)(-5, 5, -5, 0))
        glLightfv(GL_LIGHT0, GL_AMBIENT,  (GLfloat * 4)(0.3, 0.3, 0.35, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  (GLfloat * 4)(0.6, 0.6, 0.6, 1.0))

        # rotate/translate extrinsic
        glMatrixMode(GL_MODELVIEW)
        M = np.eye(4)
        M[:3, :3] = R
        M[:3,  3] = tvec.ravel()
        glLoadMatrixd(numpy_array_to_gl_matrix(M))

        glEnable(GL_DEPTH_TEST) # enable z-buffer

        # draw 3D points of quad backprojected into the image
        glColor3f(1, 0, 0)
        pyglet.graphics.draw(
            len(points_plane3d), GL_LINE_LOOP,
            ('v3f', points_plane3d.ravel()),
        )

        # turn lights on
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)

        # draw object
        monkey.draw()

        glDisable(GL_LIGHT0)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)




pyglet.app.run()
