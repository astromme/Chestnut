#include "FluidView.h"

#include <QDebug>
#include <qpainter.h>

FluidView::FluidView(int rows, int columns, QWidget* parent)
  //: QGLWidget(QGLFormat(QGL::SampleBuffers), parent),
  : QGLWidget(parent),
    m_rows(rows),
    m_cols(columns)
{
  resize(rows, columns); // each is a pixel
  
  m_fluid = MatrixXf::Zero(m_rows, m_cols);
}

FluidView::~FluidView()
{
}

void FluidView::updateFluid(float* fluidArray)
{
  for(int row=0; row<m_rows; row++) {
    for (int col=0; col<m_cols; col++) {
      // calculate the spot in the 2d array that we want.
      m_fluid(row,col) = fluidArray[(row * m_cols) + col];
    }
  }
}


void FluidView::paintEvent(QPaintEvent* e)
{
  QPainter p(this);
  QColor c;
  
  for (int row=0; row<m_rows; row++) {
    for (int col=0; col<m_cols; col++) {
      c = QColor::fromHsl(100, 255*m_fluid(row,col), 100);
      p.setPen(c);
      p.drawPoint(row, col);
    }
  }
}
