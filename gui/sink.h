#ifndef CHESTNUT_SINK_H
#define CHESTNUT_SINK_H

#include <QGraphicsObject>

#include "data.h"

class Object;
class Connection;

class Sink : public QGraphicsObject {
  public:
    enum { Type = UserType + 5 };
    int type() const;
    
    Sink(Data::Types allowedTypes, Object *parent);
    Data::Types allowedTypes() const;
    
    void setConnection(Connection *connection);
    Connection* connection() const;
    QPointF connectedCenter();
    
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
  private:
    Data::Types m_allowedTypes;
    Connection *m_connection;
    qreal m_internalMargin;
    qreal m_width;
    qreal m_height;
    Data::Type m_connectionType;
};

#endif //CHESTNUT_SINK_H