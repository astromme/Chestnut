#include "sort.h"
#include "source.h"
#include "sink.h"
#include "standardoperation.h"
#include "data.h"

#include <QDebug>

Sort::Sort(QGraphicsObject* parent)
  : Function("sort", parent)
{
  setHasOperation(false);
  setHasInputs(true);
  setHasOutputs(true);
  
  Sink *input1 = new Sink(Data::DataBlock, this);
  addSink(input1);

  Source *output1 = new Source(Data::DataBlock, this);
  addSource(output1);
}

int Sort::type() const
{
  return Type;
}

ProgramStrings Sort::flatten() const
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

  QString functioncall = QString("%1 = sort(%2, %3);")
    .arg(m_sources[0]->connectedData()[0]->name())
    .arg(m_sinks[0]->sourceData()->name())
    .arg("<");
  
  prog.second.append(functioncall);
  
  foreach(Source *source, sources()){
    QList<Data*> sourceData = source->connectedData();
    foreach (Data* sData, sourceData){
      prog = prog + sData->flatten();
    }
  }

  return prog;
}  
