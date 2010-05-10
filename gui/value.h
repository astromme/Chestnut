
#ifndef CHESTNUT_VALUE_H
#define CHESTNUT_VALUE_H

#include <QGraphicsObject>

#include "data.h"

namespace Ui {
  class ValueProperties;
}

/**
 * This class represents a scalar value in Chestnut code. It enforces
 * either a integer or a floating point type and can take a source/gives
 * a sink.
 */
class Value : public Data {
  Q_OBJECT
  public:
    Value(const QString &name, const QString &datatype="int");
    virtual ~Value();
    
    enum { Type = ChestnutItemType::Value };
    int type() const;

    virtual ProgramStrings flatten() const;
    virtual QString expression() const;
    
    virtual void mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event);
    
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    
  private slots:
    void configAccepted();
    void configRejected();
    
  private:
    QString m_name;
    int m_intValue;
    float m_floatValue;
    Ui::ValueProperties* m_ui;
};

#endif //CHESTNUT_VALUE_H
