#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gl
import pycuda.gl.autoinit
import atexit
from pycuda.tools import make_default_context
from pycuda.compiler import SourceModule
import numpy, math
from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.raw import GL
import sys


class DisplayWindow(QGLWidget):
    '''
    Widget for drawing two spirals.
    '''

    def __init__(self, parent, width, height):
        super(DisplayWindow, self).__init__(QtOpenGL.QGLFormat(QtOpenGL.QGL.SampleBuffers), parent)

        self.width = width
        self.height = height
        self.setAutoFillBackground(False)
        self.setMinimumSize(self.width, self.height)

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
        # Source parameter is NULL, Data is coming from a PBO, not host memory
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_BGRA, GL_UNSIGNED_BYTE, None)

        self.resizeGL(self.width, self.height)

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
        print 'init'

        glClearColor(0.5, 0.5, 0.5, 0.0) # Specify background as dark gray
        glEnable(GL_TEXTURE_2D)  # Enable Texturing

        depth = 4
        bufferSize = self.width*self.height*depth

        self.buffer_id = GL.GLuint(glGenBuffers(1))   # Generate a buffer ID
        self.texture_id = GL.GLuint(glGenTextures(1))  # Generate a texture ID

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.buffer_id)  # Make this the current UNPACK buffer
        glBindTexture(GL_TEXTURE_2D, self.texture_id)  # Make this the current texture

        glBufferData(GL_PIXEL_UNPACK_BUFFER, bufferSize, None, GL_DYNAMIC_COPY)  # Allocate data for the buffer. 4-channel 8-bit image
        self.cuda_buffer = pycuda.gl.RegisteredBuffer(self.buffer_id.value)

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

        self.setWindowTitle("PyCUDA Example Widget")

        self.widget = QtGui.QWidget()
        self.layout = QtGui.QHBoxLayout()

        self.codeWindow = QtGui.QPlainTextEdit()
        self.codeWindow.setMinimumSize(500, 500)
        self.runButton = QtGui.QPushButton("Run")
        self.resultsWindow = DisplayWindow(self, 500, 500)

        self.layout.addWidget(self.codeWindow)
        self.layout.addWidget(self.runButton)
        self.layout.addWidget(self.resultsWindow)

        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

        self.runButton.pressed.connect(self.update_code)

        self.timestep = 0
        self.time = QtCore.QTime()
        self.time.start()
        self.last_time = 0

        self.func = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(40) # every 40 ms means 25 fps

        self.codeWindow.setPlainText(
"""
  __global__ void display_example(unsigned char *display, int current_time)
    {
        int idx = blockIdx.x + threadIdx.x*500;

        int _x = blockIdx.x;
        int _y = threadIdx.x;

        float _width = 500;
        float _height = 500;

        float x_offset = _x - (float)_width/2;
        float y_offset = _y - (float)_height/2;

        float distance = sqrtf(x_offset * x_offset + y_offset * y_offset);

        int shade = 128 + (127 * cos(distance/10 - current_time/float(4))) / (distance/10 + 1);

        display[idx*4+0] = shade;
        display[idx*4+1] = shade;
        display[idx*4+2] = shade;
        display[idx*4+3] = 255;
    }
""")


    def update_code(self):
        try:
            self.source = SourceModule(self.codeWindow.toPlainText())

            self.func = self.source.get_function("display_example")
            self.func.prepare("PI")
        except Exception, e:
            print e
            self.func = None

        self.timestep = 0

    def update(self):
        if not self.func:
            return

        mapped_data = self.resultsWindow.cuda_buffer.map() # Map again so we can write
        dev_pointer, size = mapped_data.device_ptr_and_size()

        try:
            timer = self.func.prepared_timed_call((500, 1), (500,1, 1), dev_pointer, self.timestep)
            timer()
        except Exception, e:
            print e

        mapped_data.unmap()
        self.resultsWindow.update()

        self.timestep += 1


if __name__ == '__main__':
    app = QtGui.QApplication(['Chestnut Demo'])
    window = Demo()
    window.show()

    app.exec_()
