#include "reduce.h"
#include "source.h"
#include "sink.h"
#include "standardoperation.h"
#include "data.h"

#include <QDebug>

Reduce::Reduce(QGraphicsObject* parent)
  : Function("reduce", parent)
{
  setHasOperation(true);
  setHasInputs(true);
  setHasOutputs(true);
    
  Sink *input1 = new Sink(Data::DataBlock, this);
  addSink(input1);

  Source *output1 = new Source(Data::Value, this);
  addSource(output1);
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
 
  ProgramStrings prog;
  
  foreach(Sink *sink, sinks()){
    Data* sinkData = sink->sourceData();
    //ps += sinkData->flatten();
    prog = prog + sinkData->flatten();
  }

  QString functioncall = QString("%1 = reduce(%2, %3);")
    .arg(m_sources[0]->connectedData()[0]->name())
    .arg(operation()->name())
    .arg(m_sinks[0]->sourceData()->name());
  
  prog.second.append(functioncall);
  
  foreach(Source *source, sources()){
    QList<Data*> sourceData = source->connectedData();
    foreach (Data* sData, sourceData){
      prog = prog + sData->flatten();
    }
  }

  return prog
;
}  