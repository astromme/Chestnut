#include "value.h"

#include "sizes.h"
#include "drawingutils.h"
#include "source.h"

#include <QPainter>
#include <QApplication>

using namespace Chestnut;

Value::Value(const QString& name)
  : Data(name)
{
  m_width = 60;
  m_height = 60;
  m_name = name;
  
  Source *outputValue = new Source(Data::Value, this);
  m_sources.append(outputValue);
  
  qreal xpos = m_width/2 - outputWidth/2;
  outputValue->setPos(QPointF(xpos, m_height));
}

Value::~Value()
{

}

QRectF Value::boundingRect() const
{
  QPointF margin(1, 1);
  return QRectF(QPointF(0, 0) - margin,
                QPointF(m_width, m_height) + margin);
}

void Value::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
  
  QPointF center = QPointF(m_width/2, m_height/2);
  painter->drawPath(triangle(center, m_width, m_height));
  
  // Layout and draw text
  qreal xpos = m_width/2;
  qreal ypos = m_height;
  
  xpos -= 0.5*QApplication::fontMetrics().width(m_name);
  ypos -= 5;
  painter->drawText(xpos, ypos, m_name);
}
