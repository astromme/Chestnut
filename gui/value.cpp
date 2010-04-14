#include "value.h"

#include <QPainter>

Value::Value(QGraphicsObject* parent)
  : QGraphicsObject(parent)
{
  m_width = 20;
  m_height = 20;
}

Value::~Value()
{

}

QRectF Value::boundingRect() const
{
  qreal margin = 1;
  return QRectF(QPointF(x()-margin-m_width/2, y()-margin-m_height/2),
                QPointF(x()+margin+m_width/2, y()+margin+m_height/2));
}

void Value::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
  QPointF p1, p2, p3;
  p1 = QPointF(x(), y()-m_height/2);
  p2 = QPointF(x()-m_width/2, y()+m_height/2);
  p3 = QPointF(x()+m_width/2, y()+m_height/2);
  
  painter->drawLine(p1, p2);
  painter->drawLine(p2, p3);
  painter->drawLine(p3, p1);
  
  painter->drawText(x(), y(), "1");
}
