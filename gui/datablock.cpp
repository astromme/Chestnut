#include "datablock.h"

#include "sizes.h"
#include "data.h"
#include <QPainter>

#include <QDebug>
#include "source.h"
#include "sink.h"
#include <QApplication>

using namespace Chestnut;

DataBlock::DataBlock( const QString& name, const QString& datatype, int rows, int columns)
  : Data(name, Data::DataBlock, datatype)
{
  m_rows = rows;
  m_columns = columns;
  m_dimension = 2;
  

  Sink *in = new Sink(Data::DataBlock, this);
  in->setPos(rect().left()+rect().width()/2, rect().top());
  m_sinks.append(in);
  
  Source *out = new Source(Data::DataBlock, this);
  out->setPos(rect().left()+rect().width()/2, rect().bottom());
  m_sources.append(out);
}

DataBlock::~DataBlock()
{

}

int DataBlock::rows() const
{
  return m_rows;
}
int DataBlock::columns() const
{
  return m_columns;
}

ProgramStrings DataBlock::flatten() const
{
 
  if (isVisited()){
    return ProgramStrings();
  }
  setVisited(true);
  
  ProgramStrings ps;
  foreach(Sink *sink, sinks()){
    if (sink->isConnected()) {
      Data* sinkData = sink->sourceData();
      //ps += sinkData->flatten();
      ps = ps + sinkData->flatten();
    }
  }

  QString datablockInChestnut = "vector";	
  QString declaration = datatype() + " " + name() + " " + datablockInChestnut + ";";
  /*QString declaration = datatype() + " " +
    name() + " " +*/
    
  ps.first.append(declaration);
  
  foreach(Source *source, sources()){
    QList<Data*> sourceData = source->connectedData();
    foreach (Data* sData, sourceData){
      ps = ps + sData->flatten();
    }
  }
  
  return ps;
}

QRectF DataBlock::rect() const
{
  return QRectF(QPointF(0, 0), QSizeF(Size::dataBlockWidth, Size::dataBlockHeight));
}

QRectF DataBlock::boundingRect() const
{
  return rect().adjusted(-1, -1, 1, 1);
}

void DataBlock::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
  painter->drawRect(rect());
  painter->drawText(rect(), Qt::AlignBottom | Qt::AlignHCenter, name());
  painter->drawText(rect(), Qt::AlignTop | Qt::AlignHCenter, QString("%1 rows").arg(rows()));
  QRectF smaller = rect().adjusted(0, QApplication::fontMetrics().height(), 0, 0);
  painter->drawText(smaller, Qt::AlignTop | Qt::AlignHCenter, QString("%1 cols").arg(columns()));
}
