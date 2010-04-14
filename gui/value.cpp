#include "value.h"

#include <QPainter>
#include <QApplication>

Value::Value(const QString& name, QGraphicsObject* parent)
  : QGraphicsObject(parent)
{
  m_width = 60;
  m_height = 60;
  m_name = name;
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
  
  // Draw Triangle
  painter->drawLine(p1, p2);
  painter->drawLine(p2, p3);
  painter->drawLine(p3, p1);
  
  // Layout and draw text
  qreal xpos = x();
  qreal ypos = p3.y();
  
  xpos -= 0.5*QApplication::fontMetrics().width(m_name);
  ypos -= 5;
  painter->drawText(xpos, ypos, m_name);
}
