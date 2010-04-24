#ifndef CHESTNUT_SINK_H
#define CHESTNUT_SINK_H

#include <QGraphicsObject>

#include "data.h"

class Object;
class Connection;
class Function;

class Sink : public QGraphicsObject {
  Q_OBJECT
  public:
    enum { Type = UserType + 5 };
    int type() const;
    
    Sink(Data::Types allowedTypes, Object *parent);
    Data::Types allowedTypes() const;
    
    Object* parentObject() const;
    
    /** @returns the connected data that is providing the source for this sink.
        if this is unconnected, @returns 0
        if this is connected to a function, @returns 0
     */
    Data* sourceData() const;
    
    /** @returns the connected function that is providing the source for this sink.
        if this is unconnected, @returns 0
        if this is connected to data, @returns 0
     */
    Function* sourceFunction() const;
    
    /** @returns true if a source is connected to this sink */
    bool isConnected() const;
    Source* connectedSource() const;
    void setConnection(Connection *connection);
    Connection* connection() const;
    QPointF connectedCenter();

    virtual QRectF rect() const;
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);

  private slots:
    void moved();

  private:
    Data::Types m_allowedTypes;
    Connection *m_connection;
    qreal m_internalMargin;
    Data::Type m_connectionType;
    Object* m_parent;
};

#endif //CHESTNUT_SINK_H
