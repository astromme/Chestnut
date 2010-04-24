#include "datablock.h"

#include "sizes.h"

using namespace Chestnut;

DataBlock::DataBlock(const QString& name, int rows, int columns)
  : Data(name)
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
