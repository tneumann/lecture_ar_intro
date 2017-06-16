import numpy as np
from pyglet.gl import *

def numpy_array_to_gl_matrix(array):
    """
    Converts a given numpy array (1 or 2 dimensions)
    into an array that can be passed to an opengl function.
    For example:
        my_matrix = np.eye(4)
        glLoadMatrixd(numpy2gl(my_matrix))

    Works for float32, float64, int32 and int64 arrays
    """
    array = np.asarray(array)
    if array.ndim == 1:
        n = array.shape[0]
    elif array.ndim == 2:
        n = array.shape[0] * array.shape[1]
        # OpenGL requires column-major layout
        array = array.T
    else:
        raise ValueError("Can only map 1 or 2 dimensional arrays to OpenGL")

    # map numpy to OpenGL type
    type_dict = {
        'float32': GLfloat,
        'float64': GLdouble,
        'int32': GLint,
        'int64': GLint64
    }
    try:
        gl_type = type_dict[array.dtype.name]
    except KeyError:
        raise ValueError("Cannot map numpy arrays of type %s to OpenGL" \
                         % array.dtype.name)

    return (gl_type * n)(*(array.ravel().tolist()))


def numpy_to_pyglet_image(img):
    if img.ndim == 2:
        img = np.dstack((img, img, img)) # repeat channels for RGB
    return pyglet.image.ImageData(
        img.shape[1], img.shape[0], 'RGB', 
        str(np.ascontiguousarray(img).data), pitch=-img.shape[1]*3)

