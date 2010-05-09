#ifndef CHESTNUT_CONNECTION_H
#define CHESTNUT_CONNECTION_H

#include <QGraphicsObject>

#include "types.h"

class Source;
class Sink;

class Connection : public QGraphicsObject {
  Q_OBJECT
  public:
    Connection(Source *source, Sink *sink);
    Connection(Source *source);
    virtual ~Connection();
    
    enum { Type = ChestnutItemType::Connection };
    int type() const;
    
    void updateConnection();
    
    void setHighlighted(bool highlighted);
    
    bool isPartial() const;
    void setSink(Sink *sink);
    void setEndpoint(const QPointF& scenePoint);
    QPointF endpoint() const;
    
    Source* source() const;
    Sink* sink() const;
    
    QPainterPath path() const;
    QPainterPath endShape() const;
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    
    virtual void mousePressEvent(QGraphicsSceneMouseEvent* event);
    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent* event);
    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent* event);
    
  private slots:
    void updateEndpoints();
    
  private:
    Source *m_source;
    Sink *m_sink;
    bool m_highlighted;
    QPointF m_partialEndpoint;
};

#endif //CHESTNUT_CONNECTION_H
