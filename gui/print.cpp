#include "print.h"
#include "source.h"
#include "sink.h"
#include "standardoperation.h"
#include "data.h"

#include <QDebug>

Print::Print(QGraphicsObject* parent)
  : Function("print", parent)
{
  
  Data::Formats in;
  in << Data::DataBlock;
  in << Data::Value;
 
  Sink *input = new Sink(in, this);
  addSink(input);
  
  setHasOperation(false);
}

int Print::type() const
{
  return Type;
}

ProgramStrings Print::flatten() const
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

  QString functioncall = QString("print %1;")
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
