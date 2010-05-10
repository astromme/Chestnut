#ifndef CHESTNUT_SINK_H
#define CHESTNUT_SINK_H

#include <QGraphicsObject>

#include "data.h"
#include "types.h"

class Object;
class Connection;
class Function;

/**
 * A sink is an internal object that represents the input of some data
 * stream. It allows functions and data express that they can take some data
 * from other objects. Sinks only allow a single connection from a single
 * source. This source must also match the data type of the sink.
 */
class Sink : public QGraphicsObject {
  Q_OBJECT
  public:
    Sink(Data::Formats allowedFormats, Object *parent);
    Sink(Data::Format allowedFormat, Object *parent); /**< Convenience function with 1 allowed format */
    ~Sink();
    Data::Formats allowedFormats() const;
    
    enum { Type = ChestnutItemType::Sink };
    int type() const;
    
    Object* parentObject() const;
    
    /** @return the connected data that is providing the source for this sink.
        if this is unconnected, returns 0
        if this is connected to a function, returns 0
     */
    Data* sourceData() const;
    
    /** @return the connected function that is providing the source for this sink.
        if this is unconnected, returns 0
        if this is connected to data, returns 0
     */
    Function* sourceFunction() const;
    
    /** @return @p true if a source is connected to this sink */
    bool isConnected() const;
    
    Source* connectedSource() const;
    void setConnection(Connection *connection);
    Connection* connection() const;
    QPointF connectedCenter();

    virtual QRectF rect() const;
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);

  private:
    Data::Formats m_allowedFormats;
    Connection *m_connection;
    qreal m_internalMargin;
    Data::Format m_connectionFormat;
    Object* m_parent;
};

#endif //CHESTNUT_SINK_H
