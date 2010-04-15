#ifndef CHESTNUT_OBJECT_H
#define CHESTNUT_OBJECT_H

#include <QGraphicsObject>

class Source;
class Sink;

class Object : public QGraphicsObject {
  public:
    Object(QGraphicsObject* parent = 0);
    
    QList<Source*> sources() const;
    QList<Sink*> sinks() const;
    
    void setSources(QList<Source*> sources);
    void setSinks(QList<Sink*> sinks);
  protected:
    QList<Source*> m_sources;
    QList<Sink*> m_sinks;
};

#endif //CHESTNUT_OBJECT_H