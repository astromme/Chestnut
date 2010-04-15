#ifndef CHESTNUT_OBJECT_H
#define CHESTNUT_OBJECT_H

#include <QGraphicsItem>

class Source;
class Sink;

class Object : public QGraphicsItem {
    
  public:
    QList<Source*> sources() const;
    QList<Sink*> sinks() const;
  private:
    QList<Source*> m_sources;
    QList<Sink*> m_sinks;
};

#endif //CHESTNUT_OBJECT_H