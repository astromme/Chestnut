#include "standardoperation.h"
#include <QPainter>

StandardOperation::StandardOperation(StandardOperation::StandardOp type, Object* parent)
  : Operation(parent)
{
  m_operationType = type;
  m_radius = 10;
}

StandardOperation::~StandardOperation()
{

}

QString StandardOperation::flatten() const
{

}

QRectF StandardOperation::boundingRect() const
{
  QPointF topLeft(0, 0);
  QPointF bottomRight = QPointF(2*m_radius, 2*m_radius) + QPointF(1, 1);
  return QRectF(topLeft, bottomRight);
}
void StandardOperation::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
  painter->drawEllipse(QPointF(m_radius, m_radius), m_radius, m_radius);
  painter->drawText(boundingRect(), Qt::AlignCenter, "+");
}
