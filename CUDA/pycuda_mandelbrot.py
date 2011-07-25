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

    width = 1000
    height = 500

    def __init__(self, parent):
        QGLWidget.__init__(self, parent)
        self.resize(1000, 500)
        self.glInit()

    def closeEvent(self, event):
        glDeleteBuffers(1, self.buffer_id)
        glDeleteTextures(1, self.texture_id)

    def paintGL(self):
        '''
        Drawing routine
        '''

        #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        #glLoadIdentity()
        #glFlush()

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
        self.widget = DisplayWindow(self)
        self.setCentralWidget(self.widget)

if __name__ == '__main__':
    app = QtGui.QApplication(['Chestnut Demo'])
    window = Demo()
    window.show()

    mod = SourceModule("""
    __global__ void display_example(unsigned char *display)
    {
        int idx = blockIdx.x + blockIdx.y*%(width)s;

        int _x = blockIdx.x;
        int _y = blockIdx.y;

        int _width = %(width)s;
        int _height = %(height)s;

        float xleft = -2.5;
        float xright = 1;
        float ytop = 1;
        float ybottom = -1;


        float x0 = ((((float)_x / _width) * (xright - xleft)) + xleft);
        float y0 = ((((float)_y / _height) * (ytop - ybottom)) + ybottom);

        float x;
        x = 0;
        float y;
        y = 0;

        int iteration = 0;
        int max_iteration = 1000;

        while (((((x * x) + (y * y)) <= (2 * 2)) && (iteration < max_iteration))) {
            float xtemp;
            xtemp = (((x * x) - (y * y)) + x0);
            y = ((2 * (x * y)) + y0);
            x = xtemp;
            iteration = (iteration + 1);
        }

        if ((iteration == max_iteration)) {
            iteration = 0;
        }

        display[idx*4+0] = iteration;
        display[idx*4+1] = iteration;
        display[idx*4+2] = iteration;
        display[idx*4+3] = 255;
    }
    """ % { 'width' : 1000, 'height' : 500 })


    window.widget.mapped_data = window.widget.cuda_buffer.map() # Map again so we can write
    dev_pointer, size = window.widget.mapped_data.device_ptr_and_size()

    print dev_pointer
    print size
    func = mod.get_function("display_example")
    func.prepare("P", block=(1, 1, 1))
    func.prepared_call((1000, 500), dev_pointer)

    window.widget.mapped_data.unmap()

    app.exec_()




