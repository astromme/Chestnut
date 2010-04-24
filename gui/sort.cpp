#include "sort.h"
#include "source.h"
#include "sink.h"
#include "standardoperation.h"
#include "data.h"

#include <QDebug>

Sort::Sort(QGraphicsObject* parent)
  : Function("sort", parent)
{
  Sink *input1 = new Sink(Data::DataBlock, this);
  addSink(input1);

  Source *output1 = new Source(Data::DataBlock, this);
  addSource(output1);
  
  setHasOperation(false);
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
 
  ProgramStrings ps;
  
  foreach(Sink *sink, sinks()){
    Data* sinkData = sink->sourceData();
    //ps += sinkData->flatten();
    ps = ps + sinkData->flatten();
  }

  QString functioncall = QString("%1 = sort(%2);")
    .arg(m_sources[0]->connectedData()[0]->name())
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
