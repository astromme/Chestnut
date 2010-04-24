#ifndef CHESTNUT_OBJECT_H
#define CHESTNUT_OBJECT_H

#include <QGraphicsObject>
#include "datautils.h"

class Source;
class Sink;

class Object : public QGraphicsObject {
  Q_OBJECT
  public:
    Object(QGraphicsObject* parent = 0);
    virtual ~Object();
    
    virtual ProgramStrings flatten() const = 0;
    
    QList<Source*> sources() const;
    QList<Sink*> sinks() const;
    
    virtual void mousePressEvent ( QGraphicsSceneMouseEvent* event );
    virtual void mouseMoveEvent ( QGraphicsSceneMouseEvent* event );
    virtual void mouseReleaseEvent ( QGraphicsSceneMouseEvent* event );
    
    virtual QVariant itemChange ( GraphicsItemChange change, const QVariant& value );
    
    bool visited() const;
    void setVisited(bool visited) const;
    
  protected:
    QList<Source*> m_sources;
    QList<Sink*> m_sinks;
  private:
    bool m_moved;
    bool m_visited;
};

#endif //CHESTNUT_OBJECT_H
