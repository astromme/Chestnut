#include "drawingutils.h"

#include <QPainterPath>

void drawTriangle(QPainter* p, const QPointF &center, qreal width, qreal height)
{
  QPointF top, bottomLeft, bottomRight;
  top = center + QPointF(0, -height/2);
  bottomLeft = center + QPointF(-width/2, height/2);
  bottomRight = center + QPointF(width/2, height/2);
  
  QPainterPath path;
  // Draw Triangle
  path.moveTo(top);
  path.lineTo(bottomLeft);
  path.lineTo(bottomRight);
  path.lineTo(top);
  
  p->drawPath(path);
}
