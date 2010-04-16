#include "map.h"
#include "source.h"
#include "sink.h"


Map::Map(QGraphicsObject* parent)
  : Function("map", parent)
{
  Data::Types in1;
  in1 << Data::Value;
 
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
  
  setHasOperation(true);
  setOperation();
}

QList< QString > Map::flatten() const
{
  QList<QString> operatorString = 
}

