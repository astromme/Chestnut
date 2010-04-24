#include "value.h"

#include "sizes.h"
#include "drawingutils.h"
#include "source.h"
#include "sink.h"

#include <QPainter>
#include <QApplication>
#include <QDebug>
#include "sink.h"

using namespace Chestnut;

Value::Value( const QString& name, const QString& datatype)
  : Data(name, Data::Value, datatype)
{
  m_name = name;
  m_intValue = 0;
  m_floatValue = 0;
  
  Sink *inputValue = new Sink(Data::Value, this);
  m_sinks.append(inputValue);
  inputValue->setPos(Size::valueWidth/2 - Size::inputWidth/2, 0);
  
  Source *outputValue = new Source(Data::Value, this);
  m_sources.append(outputValue);
  
  qreal xpos = Size::valueWidth/2 - Size::outputWidth/2;
  outputValue->setPos(QPointF(xpos, Size::valueHeight));
}

Value::~Value()
{

}

int Value::type() const
{
  return Type;
}

ProgramStrings Value::flatten() const
{
  if (isVisited()){
    return ProgramStrings();
  }
  setVisited(true);

  ProgramStrings prog;
  foreach(Sink *sink, sinks()){
    if (sink->isConnected()) {
      Data* sinkData = sink->sourceData();
      //ps += sinkData->flatten();
      prog = prog + sinkData->flatten();
    }
  }
 
  if (!isInitialized()){
    QString valueInChestnut = "scalar";
    QString declaration = datatype() + " " + name() + " " + valueInChestnut + ";";
    prog.first.append(declaration);
  }
  
  foreach(Source *source, sources()){
    QList<Data*> sourceData = source->connectedData();
    foreach (Data* sData, sourceData){
      prog = prog + sData->flatten();
    }
  }
  
  return prog;
}

QRectF Value::boundingRect() const
{
  QPointF margin(1, 1);
  return QRectF(QPointF(0, 0) - margin,
                QPointF(Size::valueWidth, Size::valueHeight) + margin);
}

void Value::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
  
  QPointF center = QPointF(Size::valueWidth/2, Size::valueHeight/2);
  painter->drawPath(triangle(center, Size::valueWidth, Size::valueHeight));
  
  // Layout and draw text
  qreal xpos = Size::valueWidth/2;
  qreal ypos = Size::valueHeight;
  
  xpos -= 0.5*QApplication::fontMetrics().width(m_name);
  ypos -= 5;
  painter->drawText(xpos, ypos, m_name);
  
  if (isInitialized()) {
    QString valueText;
    if (datatype() == "float") {
      valueText = QString("%L1").arg(m_floatValue, 0, 'f', 2);
    } else if (datatype() == "int") {
      valueText = QString::number(m_intValue);
    }
    painter->drawText(boundingRect(), Qt::AlignCenter, valueText);
  }
}
