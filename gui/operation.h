#ifndef CHESTNUT_OPERATION_H
#define CHESTNUT_OPERATION_H

#include <QGraphicsObject>

class Object;

class Operation : public QGraphicsObject {
  public:
    Operation(const QString &name, Object *parent);
    virtual ~Operation();
    
    QString name() const;
  protected:
    QString m_name;
};

#endif //CHESTNUT_OPERATION_H