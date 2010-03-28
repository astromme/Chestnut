
#ifndef FLUIDVIEW_H
#define FLUIDVIEW_H

#include <QtOpenGL/QGLWidget>

#include <Eigen/Core>

USING_PART_OF_NAMESPACE_EIGEN

class FluidView : public QGLWidget {
  Q_OBJECT
  public:
    FluidView(int rows, int columns, QWidget* parent = 0);
    ~FluidView();
    
    void updateFluid(float* fluidArray);
    virtual void paintEvent(QPaintEvent* e);
    
  private:
    int m_rows;
    int m_cols;
    MatrixXf m_fluid;
    
};

#endif