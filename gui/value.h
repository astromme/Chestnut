
#ifndef CHESTNUT_VALUE_H
#define CHESTNUT_VALUE_H

#include <QGraphicsObject>

#include "data.h"

class Value : public Data {
  Q_OBJECT
  public:
    Value(const QString &name, const QString &datatype="int");
    virtual ~Value();
    
    enum { Type = ChestnutItemType::Value };
    int type() const;

    virtual ProgramStrings flatten() const;
    
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
  private:
    QString m_name;
    int m_intValue;
    float m_floatValue;
};

#endif //CHESTNUT_VALUE_H
