#ifndef CHESTNUT_OBJECT_H
#define CHESTNUT_OBJECT_H

#include <QGraphicsObject>
#include "datautils.h"
#include "types.h"

class Source;
class Sink;

/**
 * This class is a generic internal representation of an object that can be
 * placed on the canvas and manipulated. It has the concept of sources and
 * sinks. This class is pure virtual and can be subclassed into things that
 * the user can use.
 *
 * @see Function
 * @see Data
 */
class Object : public QGraphicsObject {
  Q_OBJECT
  public:
    Object(QGraphicsObject* parent = 0);
    virtual ~Object();
    
    virtual ProgramStrings flatten() const = 0;
    
    QList<Source*> sources() const;
    QList<Sink*> sinks() const;
    
    virtual bool isData() const;
    virtual bool isFunction() const;
    
    virtual void mousePressEvent(QGraphicsSceneMouseEvent* event);
    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent* event);
    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent* event);

    virtual QVariant itemChange(GraphicsItemChange change, const QVariant& value);
    
    bool isVisited() const;
    void setVisited(bool visited) const;
    
  protected:
    QList<Source*> m_sources;
    QList<Sink*> m_sinks;
  private:
    bool m_moved;
    bool m_visited;
};

#endif //CHESTNUT_OBJECT_H
