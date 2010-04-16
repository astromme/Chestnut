#ifndef CHESTNUT_DATABLOCK_H
#define CHESTNUT_DATABLOCK_H

#include <QGraphicsObject>

#include "data.h"

class DataBlock : public Data {
  public:
    DataBlock(const QString &name);
    virtual ~DataBlock();
    
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
  private:
    int m_dimension;
};

#endif //CHESTNUT_DATABLOCK_H