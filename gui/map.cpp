#include "map.h"
#include "source.h"
#include "sink.h"
#include "standardoperation.h"
#include "data.h"

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
  addSink(input1);
  Sink *input2 = new Sink(in2, this);
  addSink(input2);

  Source *output1 = new Source(Data::DataBlock, this);
  addSource(output1);
  
  Operation *op = new StandardOperation(StandardOperation::Multiply, this);
  setHasOperation(true);
  setOperation(op);
}

ProgramStrings Map::flatten() const
{
  //TODO Finish/talk to ryan
  //TODO Make flatten return ProgramStrings
  //qDebug() << m_sources[0]
  foreach(Sink *connectedSink, m_sources[0]->connectedSinks()) {
    qDebug() << "connected sink";
    Object *connectedObject = connectedSink->parentObject();
    
    QString functionLine = QString("%1 = map(%2, %3, %4);")
      .arg(connectedObject->name())
      .arg(operation()->name())
      .arg(m_sinks[1]->connectedSource()->parentObject()->name())
      .arg(m_sinks[0]->connectedSource()->parentObject()->name());
      
    ProgramStrings ps;
    ps.second.append(functionLine);
    ProgramStrings connected = connectedObject->flatten();
    return ProgramStrings(ps.first + ps.first, ps.second + connected.second);
    //return ps + connectedObject->flatten();
  }
  Data::tempData(Data::DataBlock);
  return ProgramStrings();
  //map(+,2,data);
}