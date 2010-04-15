#ifndef CHESTNUT_CONNECTION_H
#define CHESTNUT_CONNECTION_H

#include <QGraphicsItem>

class Source;
class Sink;

class Connection : public QGraphicsItem {
  public:
    Connection(Source *source, Sink *sink);
    Connection* partialConnection(Source *source);
    
    bool isPartial() const;
    void setSink(Sink *sink);
    void setEndpoint(const QPointF &point);
    QPointF endpoint();
    
    Source* source() const;
    Sink* sink() const;
    
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    
  private:
    Source *m_source;
    Sink *m_sink;
    QPointF m_partialEndpoint;
};

#endif //CHESTNUT_CONNECTION_H