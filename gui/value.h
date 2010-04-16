#ifndef CHESTNUT_VALUE_H
#define CHESTNUT_VALUE_H

#include <QGraphicsObject>

#include "data.h"

class Value : public Data {
  public:
    Value(const QString &name);
    virtual ~Value();
    
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
  private:
    qreal m_width;
    qreal m_height;
    QString m_name;
};

#endif //CHESTNUT_VALUE_H