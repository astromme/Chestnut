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
  m_name = name;
  
  Source *outputValue = new Source(Data::Value, this);
  m_sources.append(outputValue);
  
  qreal xpos = valueWidth/2 - outputWidth/2;
  outputValue->setPos(QPointF(xpos, valueHeight));
}

Value::~Value()
{

}

QRectF Value::boundingRect() const
{
  QPointF margin(1, 1);
  return QRectF(QPointF(0, 0) - margin,
                QPointF(valueWidth, valueHeight) + margin);
}

void Value::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
  
  QPointF center = QPointF(valueWidth/2, valueHeight/2);
  painter->drawPath(triangle(center, valueWidth, valueHeight));
  
  // Layout and draw text
  qreal xpos = valueWidth/2;
  qreal ypos = valueHeight;
  
  xpos -= 0.5*QApplication::fontMetrics().width(m_name);
  ypos -= 5;
  painter->drawText(xpos, ypos, m_name);
}
