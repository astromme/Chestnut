#include "reduce.h"
#include "source.h"
#include "sink.h"
#include "standardoperation.h"
#include "data.h"

#include <QDebug>

Reduce::Reduce(QGraphicsObject* parent)
  : Function("reduce", parent)
{
  Sink *input1 = new Sink(Data::DataBlock, this);
  addSink(input1);

  Source *output1 = new Source(Data::Value, this);
  addSource(output1);
  
  Operation *op = new StandardOperation(StandardOperation::Add, this); // TODO allow op to be set
  setHasOperation(true);
  setOperation(op);
}

int Reduce::type() const
{
  return Type;
}

ProgramStrings Reduce::flatten() const
{
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

  QString functioncall = QString("%1 = reduce(%2, %3);")
    .arg(m_sources[0]->connectedData()[0]->name())
    .arg(operation()->name())
    .arg(m_sinks[0]->sourceData()->name());
  
  ps.second.append(functioncall);
  
  foreach(Source *source, sources()){
    QList<Data*> sourceData = source->connectedData();
    foreach (Data* sData, sourceData){
      ps = ps + sData->flatten();
    }
  }

  return ps;
}  