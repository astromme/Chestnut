#ifndef CHESTNUT_VALUE_H
#define CHESTNUT_VALUE_H

#include <QGraphicsObject>

#include "input.h"
#include "output.h"

class Value : public QGraphicsObject, Input, Output {
  public:
    Value(QGraphicsObject *parent=0);
    virtual ~Value();
    
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
  private:
    qreal m_width;
    qreal m_height;
};

#endif //CHESTNUT_VALUE_H