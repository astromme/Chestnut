#include "map.h"
#include "source.h"
#include "sink.h"
#include "standardoperation.h"

#include <QDebug>

Map::Map(QGraphicsObject* parent)
  : Function("map", parent)
{
  Data::Types in1;
  in1 << Data::DataBlock;
 
  Data::Types in2;
  in2 << Data::DataBlock;
  in2 << Data::Value;
  
  Sink *input1 = new Sink(in1, this);
  Sink *input2 = new Sink(in2, this);
  Source *output1 = new Source(Data::DataBlock, this);
  m_sinks.append(input1);
  m_sinks.append(input2);
  m_sources.append(output1);
  
  QPointF margin(1, 1);
  QPointF i1Pos = inputsRect().topLeft() + margin;
  i1Pos += QPointF(2, 2); // internal margin
  input1->setPos(i1Pos);
  QPointF i2Pos = i1Pos + QPointF(2+input1->boundingRect().width(), 0);
  input2->setPos(i2Pos);
  
  QPointF o1Pos = outputsRect().topLeft();
  output1->setPos(o1Pos);
  
  setHasOperation(true);
  setOperation(new StandardOperation(StandardOperation::Multiply, this));
}

QStringList Map::flatten() const
{
  //TODO Finish/talk to ryan
  //qDebug() << m_sources[0]
  foreach(Sink *connectedSink, m_sources[0]->connectedSinks()) {
    qDebug() << "connected sink";
    Object *connectedObject = connectedSink->parentObject();
    QString functionLine = QString("%1 = %2(%3, %4, %5);").arg(connectedObject->name())
                                                          .arg(m_name)
                                                          .arg(operation()->name())
                                                          .arg(m_sinks[0]->connectedSource()->parentObject()->name())
                                                          .arg(m_sinks[1]->connectedSource()->parentObject()->name());
    QStringList l;
    l.append(functionLine);
    return l + connectedObject->flatten();
  }
  return QStringList();
  //map(+,2,data);
}

