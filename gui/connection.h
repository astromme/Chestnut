#ifndef CHESTNUT_CONNECTION_H
#define CHESTNUT_CONNECTION_H

#include <QGraphicsObject>

#include "types.h"

class Source;
class Sink;

/**
 * A connection is an internal class that connects a single source to a single
 * sink. It should only be created by the respective source or sink and will
 * not work correctly if manually initialized.
 * 
 * Connections show up visually on the canvas as curved lines from sources
 * to sinks.
 */
class Connection : public QGraphicsObject {
  Q_OBJECT
  public:
    /** Create a connection that links the given source to the given sink */
    Connection(Source *source, Sink *sink);
    /** Create a partial connection that stems from the given source. */
    Connection(Source *source);
    virtual ~Connection();
    
    // Type is used for internal QGraphicsItem type() options
    enum { Type = ChestnutItemType::Connection };
    int type() const;
    
    /**
     * Highlights the endpoint of the connection. Useful for showing
     * the user when it is appropriate to let go of the mouse, connecting
     * the connection to a sink
     *
     * @param highlighted enable or disable the highlighting of this connection.
     *
     */
    void setHighlighted(bool highlighted);
    
    /**
     * A connection is only valid when both endpoints are connected. Therefore,
     * a partial connection is a useful concept. A connection might be partial
     * because it is still being dragged by the user to the endpoint sink.
     *
     * @return @c true if the connection has no sink endpoint, @c false otherwise
     */
    bool isPartial() const;
    
    /**
     * Directly sets the sink of this connection. Unconnects the old sink
     * if it existed and sets up this new one correctly
     *
     * @param sink the sink that is the new endpoint of this connection. Pass 0 to make this a partial connection
     */
    void setSink(Sink *sink);
    
    /**
     * When the sink is only partially connected, sets the point on the scene
     * where the endpoint should be located.
     *
     * @param scenePoint the point in scene coordinates where the endpoint should be located
     */
    void setEndpoint(const QPointF& scenePoint);
    
    /**
     * @return the current endpoint in scene coordinates. Only valid when the connection is not partial
     */
    QPointF endpoint() const;
    
    /**
     * @return the source of the connection, 0 if there is no source
     */
    Source* source() const;
    
    /**
     * @return the sink of the connection, 0 if partially connected
     */
    Sink* sink() const;
    
    /**
     * @return the graphical path that represents the connection
     */
    QPainterPath path() const;
    /**
     * @return the graphical shape that is located on top of the
     *         endpoint of the connection
     */
    QPainterPath endShape() const;
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    
    virtual void mousePressEvent(QGraphicsSceneMouseEvent* event);
    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent* event);
    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent* event);
    
  private slots:
    /**
     * Update the connection's graphical representation. Should be used
     * when the geometry changes
     */
    void updateEndpoints();
    
  private:
    Source *m_source;
    Sink *m_sink;
    bool m_highlighted;
    QPointF m_partialEndpoint;
};

#endif //CHESTNUT_CONNECTION_H
