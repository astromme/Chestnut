#!/usr/bin/env python


#############################################################################
##
## Copyright (C) 2010 Riverbank Computing Limited.
## Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
## All rights reserved.
##
## This file is part of the examples of PyQt.
##
## $QT_BEGIN_LICENSE:BSD$
## You may use this file under the terms of the BSD license as follows:
##
## "Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
##   * Redistributions of source code must retain the above copyright
##     notice, this list of conditions and the following disclaimer.
##   * Redistributions in binary form must reproduce the above copyright
##     notice, this list of conditions and the following disclaimer in
##     the documentation and/or other materials provided with the
##     distribution.
##   * Neither the name of Nokia Corporation and its Subsidiary(-ies) nor
##     the names of its contributors may be used to endorse or promote
##     products derived from this software without specific prior written
##     permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
## $QT_END_LICENSE$
##
#############################################################################


import sys
import math, random

from PyQt4 import QtCore, QtGui, QtOpenGL

try:
    from OpenGL.GL import *
except ImportError:
    app = QtGui.QApplication(sys.argv)
    QtGui.QMessageBox.critical(None, "OpenGL overpainting",
            "PyOpenGL must be installed to run this example.")
    sys.exit(1)


class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        super(GLWidget, self).__init__(QtOpenGL.QGLFormat(QtOpenGL.QGL.SampleBuffers), parent)

        self.trolltechPurple = QtGui.QColor.fromCmykF(0.39, 0.39, 0.0, 0.0)

        self.animationTimer = QtCore.QTimer()
        self.animationTimer.setSingleShot(False)
        self.animationTimer.timeout.connect(self.animate)
        self.animationTimer.start(25)

        self.setAutoFillBackground(False)
        self.setMinimumSize(200, 200)
        self.setWindowTitle("Overpainting a Scene")

    def paintEvent(self, event):
        self.makeCurrent()

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        self.qglClearColor(self.trolltechPurple.dark())
        self.setupViewport(self.width(), self.height())

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        self.drawInstructions(painter)
        painter.end()

    def resizeGL(self, width, height):
        self.setupViewport(width, height)

    def sizeHint(self):
        return QtCore.QSize(400, 400)

    def normalizeAngle(self, angle):
        while angle < 0:
            angle += 360 * 16
        while angle > 360 * 16:
            angle -= 360 * 16
        return angle

    def animate(self):
        self.update()

    def setupViewport(self, width, height):
        side = min(width, height)
        glViewport((width - side) / 2, (height - side) / 2, side, side)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-0.5, +0.5, +0.5, -0.5, 4.0, 15.0)
        glMatrixMode(GL_MODELVIEW)

    def drawInstructions(self, painter):
        text = "Click and drag with the left mouse button to rotate the Qt " \
                "logo."
        metrics = QtGui.QFontMetrics(self.font())
        border = max(4, metrics.leading())

        rect = metrics.boundingRect(0, 0, self.width() - 2*border,
                int(self.height()*0.125),
                QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap, text)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.fillRect(QtCore.QRect(0, 0, self.width(), rect.height() + 2*border), QtGui.QColor(0, 0, 0, 127))
        painter.setPen(QtCore.Qt.white)
        painter.fillRect(QtCore.QRect(0, 0, self.width(), rect.height() + 2*border), QtGui.QColor(0, 0, 0, 127))
        painter.drawText((self.width() - rect.width())/2, border, rect.width(),
                rect.height(), QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap,
                text)


if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    window = GLWidget()
    window.show()
    sys.exit(app.exec_())

