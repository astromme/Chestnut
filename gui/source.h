#ifndef CHESTNUT_SOURCE_H
#define CHESTNUT_SOURCE_H

#include <QGraphicsItem>

#include "data.h"

class Connection;
class Object;
class Sink;

class Source : public QGraphicsItem {
  public:
    Source(Data::Type type, Object *parent);
    Data::Type dataType() const;
    
    Connection* connectToSink(Sink *sink);
    
    void addConnection(Connection *connection);
    void removeConnection(Connection *connection);
    void removeAllConnections();
    
    QPointF connectedCenter() const;
    
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
  private:
    Data::Type m_dataType;
    QList<Connection*> m_connections;
    qreal m_width;
    qreal m_height;
    
};

#endif //CHESTNUT_SOURCE_H