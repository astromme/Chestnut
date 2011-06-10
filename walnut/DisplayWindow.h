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

#ifndef DISPLAYWINDOW_H
#define DISPLAYWINDOW_H

#include "walnut_global.h"

#include <QtOpenGL/QGLWidget>
#include <thrust/device_vector.h>

namespace WALNUT_EXPORT ChestnutCore {

class WALNUT_EXPORT DisplayWindow : public QGLWidget
{
public:
  DisplayWindow(const QSize &windowSize, QWidget *parent=0);
  virtual ~DisplayWindow();

  thrust::device_ptr<uchar4> displayData() const;

protected:
  void initializeGL();
  void resizeGL(int w, int h);
  void paintGL();
  void mousePressEvent(QMouseEvent* mouseEvent);
  void mouseReleaseEvent(QMouseEvent* mouseEvent);
  void mouseDoubleClickEvent(QMouseEvent* mouseEvent);
  void mouseMoveEvent(QMouseEvent* mouseEvent);
  void wheelEvent(QWheelEvent* wheelEvent);

private:
  QSize m_size;
  uchar4 *m_data;
  GLuint m_bufferID;
  GLuint m_textureID;
};

} // End namespace Chestnut

#endif // QCAD_H

/*
  thrust::device_vector<uchar4> *unpadded = display.unpadded();

  thrust::copy(unpadded->begin(), unpadded->begin()+size.width()*size.height(), mappedDisplay);
  */

