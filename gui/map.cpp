#include "map.h"
#include "source.h"
#include "sink.h"
#include "standardoperation.h"
#include "data.h"

#include <QDebug>

Map::Map(QGraphicsObject* parent)
  : Function("map", parent)
{
  Data::Formats in1;
  in1 << Data::DataBlock;
 
  Data::Formats in2;
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

  if (isVisited()){
    return ProgramStrings();
  }
  
  setVisited(true);
 
  ProgramStrings ps;
  
  foreach(Sink *sink, sinks()){
    Data* sinkData = sink->sourceData();
    //ps += sinkData->flatten();
    ps = ps + sinkData->flatten();
  }

  QString functioncall = QString("%1 = map(%2, %3, %4);")
    //.arg(m_sources[0]->connectedData)
    .arg(m_sources[0]->connectedData()[0]->name())
    .arg(operation()->name())
    .arg(m_sinks[1]->sourceData()->name())
    .arg(m_sinks[0]->sourceData()->name());
    
  ps.second.append(functioncall);
  
  foreach(Source *source, sources()){
    QList<Data*> sourceData = source->connectedData();
    foreach (Data* sData, sourceData){
      ps = ps + sData->flatten();
    }
  }
  
  // qDebug() << "ProgStrings:" << ps; // TODO: remove

  return ps;
  
/* 
  foreach(Sink *connectedSink, m_sources[0]->connectedSinks()) {
    qDebug() << "connected sink";
    Object *connectedObject = connectedSink->parentObject();
   
    // if either of the sinks are coming from a DataBlock source, we need to create a temporary variable in between the two functions
    if (m_sinks[1]->connectedSource()->dataType() == Data::DataBlock){
      qDebug() << "Connected Source at location 1 is a DataBlock";
    }
    if (m_sinks[0]->connectedSource()->dataType() == Data::DataBlock){
      qDebug() << "Connected Source at location 0 is a DataBlock";
    }
    
    QString functionLine = QString("%1 = map(%2, %3, %4);")
      .arg(connectedObject->name())
      .arg(operation()->name())
      .arg(m_sinks[1]->connectedSource()->parentObject()->name())
      .arg(m_sinks[0]->connectedSource()->parentObject()->name());
      
    ProgramStrings ps;
    ps.second.append(functionLine);
   
    return ps + connectedObject->flatten();
  }
  QString tmpData = Data::tempData(Data::DataBlock);
*/
  //map(+,2,data);
}