#ifndef CHESTNUT_DATABLOCK_H
#define CHESTNUT_DATABLOCK_H

#include <QGraphicsObject>

#include "input.h"
#include "output.h"

class DataBlock : public QGraphicsObject, Input, Output {
  public:
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
  private:
    int m_dimension;
};

#endif //CHESTNUT_DATABLOCK_H