#ifndef CHESTNUT_SOURCE_H
#define CHESTNUT_SOURCE_H

#include <QGraphicsObject>

#include "data.h"

class Connection;
class Object;
class Sink;

class Source : public QGraphicsObject {
  Q_OBJECT
  public:
    Source(Data::Type type, Object *parent);
    Data::Type dataType() const;
    
    Connection* connectToSink(Sink *sink);
    
    Object* parentObject() const;
    
    QList<Sink*> connectedSinks() const;
    void addConnection(Connection *connection);
    void removeConnection(Connection *connection);
    void removeAllConnections();
    
    QPointF connectedCenter() const;
    
    virtual void mousePressEvent(QGraphicsSceneMouseEvent* event);
    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent* event);
    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent* event);
    
    virtual QRectF rect() const;
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);

private slots:
  void moved();

  private:
    Data::Type m_dataType;
    QList<Connection*> m_connections;
    Connection* m_activeConnection;
    Object* m_parent;
};

#endif //CHESTNUT_SOURCE_H
