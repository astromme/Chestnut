/*
 *   Copyright 2009 Andrew Stromme <astromme@chatonka.com>
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU Library General Public License as
 *   published by the Free Software Foundation; either version 2, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details
 *
 *   You should have received a copy of the GNU Library General Public
 *   License along with this program; if not, write to the
 *   Free Software Foundation, Inc.,
 *   51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */


#include "DisplayWindow.h"

#include <QDebug>

#include <cuda_gl_interop.h>

using namespace Walnut;

DisplayWindow::DisplayWindow(const QSize &windowSize, QWidget *parent)
 : QGLWidget(parent),
   m_size(windowSize),
   m_data(0)
{
  resize(m_size);
  setWindowTitle("Chestnut Output");
  updateGL(); // init so we can get our texture ready
}

DisplayWindow::~DisplayWindow() {
  glDeleteBuffers(1, &m_bufferID);
  glDeleteTextures(1, &m_textureID);
}

thrust::device_ptr<uchar4> DisplayWindow::displayData() const {
  return thrust::device_ptr<uchar4>(m_data);
}

void DisplayWindow::initializeGL() {
  glClearColor(0.5, 0.5, 0.5, 0.0); // Specify background as dark gray
  glEnable(GL_TEXTURE_2D);  // Enable Texturing

  int bufferSize = m_size.width()*m_size.height()*4;

  glGenBuffers(1,  &m_bufferID);   // Generate a buffer ID
  glGenTextures(1, &m_textureID);  // Generate a texture ID

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_bufferID);  // Make this the current UNPACK buffer
  glBindTexture(GL_TEXTURE_2D, m_textureID);  // Make this the current texture

  glBufferData(GL_PIXEL_UNPACK_BUFFER, bufferSize, NULL, GL_DYNAMIC_COPY);  // Allocate data for the buffer. 4-channel 8-bit image
  cudaGLRegisterBufferObject(m_bufferID);

  // Allocate the texture memory.  The last parameter is NULL since we only
  // want to allocate memory, not initialize it
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_size.width(), m_size.height(), 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
  // Must set the filter mode, GL_LINEAR enables interpolation when scaling
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  cudaGLMapBufferObject((void**)&m_data, m_bufferID);
}

void DisplayWindow::resizeGL(int w, int h) {
  glViewport(0, 0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, 1, 0, 1); // set origin to bottom left corner
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

void DisplayWindow::paintGL() {
  cudaGLUnmapBufferObject(m_bufferID);  // unmap so we can display

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_bufferID);  // Select the appropriate buffer
  glBindTexture(GL_TEXTURE_2D, m_textureID);  // Select the appropriate texture

  // Make a texture from the buffer
  // Source parameter is NULL, Data is coming from a PBO, not host memory                             here
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_size.width(), m_size.height(), GL_BGRA, GL_UNSIGNED_BYTE, NULL);

  glClear(GL_COLOR_BUFFER_BIT); // Paint background

  int z = 0;

  glBegin(GL_QUADS); {
    glTexCoord2f(0, 1.0f);
    glVertex3f(0, 0, z);
    glTexCoord2f(0, 0);
    glVertex3f(0, 1.0f, z);
    glTexCoord2f(1.0f, 0);
    glVertex3f(1.0f, 1.0f, z);
    glTexCoord2f(1.0f, 1.0f);
    glVertex3f(1.0f, 0, z);
  } glEnd();

  cudaGLMapBufferObject((void**)&m_data, m_bufferID); // Map again so we can write
}

void DisplayWindow::mousePressEvent(QMouseEvent* ) {

}

void DisplayWindow::mouseReleaseEvent(QMouseEvent* ) {

}

void DisplayWindow::mouseDoubleClickEvent(QMouseEvent* ) {

}

void DisplayWindow::mouseMoveEvent(QMouseEvent *mouseEvent) {

}

void DisplayWindow::wheelEvent(QWheelEvent* ) {

}

