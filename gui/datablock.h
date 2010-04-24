#ifndef CHESTNUT_DATABLOCK_H
#define CHESTNUT_DATABLOCK_H

#include <QGraphicsObject>

#include "data.h"

// assumed to be 2d
class DataBlock : public Data {
  public:
    DataBlock(const QString &name, const QString &datatype, int rows, int columns);
    virtual ~DataBlock();
    
    int rows() const;
    int columns() const;
    
    virtual ProgramStrings flatten() const;
    
    QRectF rect() const;
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
  private:
    int m_dimension;
    int m_rows;
    int m_columns;
};

#endif //CHESTNUT_DATABLOCK_H