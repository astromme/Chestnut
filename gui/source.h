#ifndef CHESTNUT_SOURCE_H
#define CHESTNUT_SOURCE_H

#include <QGraphicsObject>

#include "data.h"

class Connection;
class Object;
class Sink;
class Function;

/**
 * A source is an internal object that represents the output of some data
 * stream. It allows functions and data express that they have some data
 * to provide to other objects. Sources allow multiple connections to
 * other sinks as long as the type of the sink matches the type of the source.
 */
class Source : public QGraphicsObject {
  Q_OBJECT
  public:
    Source(Data::Format type, Object *parent);
    ~Source();
    Data::Format format() const;
    
    enum { Type = ChestnutItemType::Source };
    int type() const;
    
    Connection* connectToSink(Sink *sink);
    
    Object* parentObject() const;
  
    /** @returns the list of connected data objects that are taking data from this source
        if this is unconnected, @returns 0
        if this is connected to functions, @returns 0
     */
    QList<Data*> connectedData() const;
    
    /** @returns the list of connected function objects that are taking data from this source
        if this is unconnected, @returns 0
        if this is connected to data, @returns 0
     */
    QList<Function*> connectedFunctions() const;
    
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

  private:
    Data::Format m_format;
    QList<Connection*> m_connections;
    Connection* m_activeConnection;
    Object* m_parent;
};

#endif //CHESTNUT_SOURCE_H
