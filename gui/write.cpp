#include "write.h"
#include "source.h"
#include "sink.h"
#include "standardoperation.h"
#include "data.h"

#include <QDebug>

Write::Write(QGraphicsObject* parent)
  : Function("write", parent)
{
  
  Data::Formats in;
  in << Data::DataBlock;
  in << Data::Value;
 
  Sink *input = new Sink(in, this);
  addSink(input);
  
  setHasOperation(false);
}

int Write::type() const
{
  return Type;
}

QString Write::filename() const
{
  return m_filename;
}

void Write::setFilename(const QString& fname)
{
  m_filename = fname;
}

ProgramStrings Write::flatten() const
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

  QString functioncall = QString("write(%1,\"%2\");")
    .arg(m_sinks[0]->sourceData()->name())
    .arg(filename());
  
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

