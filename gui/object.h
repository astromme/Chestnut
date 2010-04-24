#ifndef CHESTNUT_OBJECT_H
#define CHESTNUT_OBJECT_H

#include <QGraphicsObject>
#include "datautils.h"

class Source;
class Sink;

class Object : public QGraphicsObject {
  Q_OBJECT
  public:
    Object(const QString &name, QGraphicsObject* parent = 0);
    virtual ~Object();
    
    virtual ProgramStrings flatten() const = 0;
    QString name() const;
    
    QList<Source*> sources() const;
    QList<Sink*> sinks() const;
    
    virtual void mousePressEvent ( QGraphicsSceneMouseEvent* event );
    virtual void mouseMoveEvent ( QGraphicsSceneMouseEvent* event );
    virtual void mouseReleaseEvent ( QGraphicsSceneMouseEvent* event );
    
    virtual QVariant itemChange ( GraphicsItemChange change, const QVariant& value );
    
    bool visited();
    setVisited(bool visited);
    
  protected:
    QList<Source*> m_sources;
    QList<Sink*> m_sinks;
    QString m_name;
private:
    bool m_moved;
    bool m_visited;
};

#endif //CHESTNUT_OBJECT_H
