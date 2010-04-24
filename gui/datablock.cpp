#include "datablock.h"

#include "sizes.h"
#include "data.h"

#include <QDebug>

using namespace Chestnut;

DataBlock::DataBlock( const QString& name, const QString& datatype, int rows, int columns)
  : Data(name, Data::DataBlock, datatype)
{
  m_rows = rows;
  m_columns = columns;
  m_dimension = 2;
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
  QString datablockInChestnut = "vector";	
  QString declaration = datatype() + " " + name() + " " + datablockInChestnut;
  qDebug() << "flatten called on datablock: " << declaration;
  ProgramStrings ps;
  ps.first.append(declaration);
  
  return ProgramStrings();
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
  
}
