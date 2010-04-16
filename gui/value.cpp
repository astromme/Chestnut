#include "value.h"

#include <QPainter>
#include <QApplication>
#include "drawingutils.h"
#include "source.h"

Value::Value(const QString& name)
  : Data(name)
{
  m_width = 60;
  m_height = 60;
  m_name = name;
  
  Source *outputValue = new Source(Data::Value, this);
  m_sources.append(outputValue);
  
  outputValue->setPos(QPointF(0, m_height/2));
}

Value::~Value()
{

}

QRectF Value::boundingRect() const
{
  qreal margin = 1;
  return QRectF(QPointF(-margin-m_width/2, -margin-m_height/2),
                QPointF(margin+m_width/2, +margin+m_height/2));
}

void Value::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
  
  painter->drawPath(triangle(QPointF(0, 0), m_width, m_height));
  
  // Layout and draw text
  qreal xpos = 0;
  qreal ypos = m_height/2;
  
  xpos -= 0.5*QApplication::fontMetrics().width(m_name);
  ypos -= 5;
  painter->drawText(xpos, ypos, m_name);
}
