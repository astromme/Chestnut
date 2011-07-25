#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gl
import pycuda.gl.autoinit
import atexit
from pycuda.tools import make_default_context
from pycuda.compiler import SourceModule
import numpy, math
import PyQt4.QtGui as QtGui
import PyQt4.QtCore as QtCore
from PyQt4.QtOpenGL import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.raw import GL


class DisplayWindow(QGLWidget):
    '''
    Widget for drawing two spirals.
    '''

    def __init__(self, parent, width, height):
        QGLWidget.__init__(self, parent)
        self.width = width
        self.height = height
        self.setMinimumSize(self.width, self.height)
        self.glInit()

    def closeEvent(self, event):
        glDeleteBuffers(1, self.buffer_id)
        glDeleteTextures(1, self.texture_id)

    def paintGL(self):
        '''
        Drawing routine
        '''

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.buffer_id)  # Select the appropriate buffer
        glBindTexture(GL_TEXTURE_2D, self.texture_id)  # Select the appropriate texture

        # Make a texture from the buffer
        # Source parameter is NULL, Data is coming from a PBO, not host memory                             here
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_BGRA, GL_UNSIGNED_BYTE, None)

        glClear(GL_COLOR_BUFFER_BIT) # Paint background

        z = 0

        glBegin(GL_QUADS)
        glTexCoord2f(0, 1.0)
        glVertex3f(0, 0, z)
        glTexCoord2f(0, 0)
        glVertex3f(0, 1.0, z)
        glTexCoord2f(1.0, 0)
        glVertex3f(1.0, 1.0, z)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(1.0, 0, z)
        glEnd()

    def resizeGL(self, w, h):
        '''
        Resize the GL window
        '''

        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, 1, 0, 1) # set origin to bottom left corner
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def initializeGL(self):
        '''
        Initialize GL
        '''
        glClearColor(0.5, 0.5, 0.5, 0.0) # Specify background as dark gray
        glEnable(GL_TEXTURE_2D)  # Enable Texturing

        depth = 4
        bufferSize = self.width*self.height*depth

        self.buffer_id = glGenBuffers(1)   # Generate a buffer ID
        self.texture_id = glGenTextures(1)  # Generate a texture ID

        print self.buffer_id
        print type(self.buffer_id)


        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.buffer_id)  # Make this the current UNPACK buffer
        glBindTexture(GL_TEXTURE_2D, self.texture_id)  # Make this the current texture

        glBufferData(GL_PIXEL_UNPACK_BUFFER, bufferSize, None, GL_DYNAMIC_COPY)  # Allocate data for the buffer. 4-channel 8-bit image
        self.cuda_buffer = pycuda.gl.RegisteredBuffer(int(self.buffer_id))

        # Allocate the texture memory.  The last parameter is NULL since we only
        # want to allocate memory, not initialize it
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.width, self.height, 0, GL_BGRA, GL_UNSIGNED_BYTE, None)
        # Must set the filter mode, GL_LINEAR enables interpolation when scaling
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

# You don't need anything below this
class Demo(QtGui.QMainWindow):
    ''' Example class for using SpiralWidget'''

    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.widget = DisplayWindow(self, 500, 500)
        self.setCentralWidget(self.widget)

if __name__ == '__main__':
    app = QtGui.QApplication(['Chestnut Demo'])
    window = Demo()
    window.show()

    mod = SourceModule("""
    __global__ void display_example(unsigned char *display, int current_time)
    {
        int idx = blockIdx.x + threadIdx.x*%(width)s;

        int _x = blockIdx.x;
        int _y = threadIdx.x;

        float _width = %(width)s;
        float _height = %(height)s;

        float x_offset = _x - _width/2;
        float y_offset = _y - _height/2;

        float distance = sqrtf(x_offset * x_offset + y_offset * y_offset);

        int shade = 128 + (127 * cos(distance/10 - current_time/7.0)) / (distance/10 + 1);

        display[idx*4+0] = shade;
        display[idx*4+1] = 150;
        display[idx*4+2] = shade;
        display[idx*4+3] = 255;
    }
    """ % { 'width' : 500, 'height' : 500 })


    window.widget.mapped_data = window.widget.cuda_buffer.map() # Map again so we can write
    dev_pointer, size = window.widget.mapped_data.device_ptr_and_size()

    func = mod.get_function("display_example")
    func.prepare("PI")
    func.prepared_call((500, 1), (500,1, 1), dev_pointer, 10)

    window.widget.mapped_data.unmap()

    app.exec_()





