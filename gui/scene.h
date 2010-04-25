#ifndef CHESTNUT_SCENE_H
#define CHESTNUT_SCENE_H

#include <QGraphicsScene>

class Scene : public QGraphicsScene {
  public:
    Scene(QObject* parent = 0);
    
  protected:
    virtual void dragEnterEvent(QGraphicsSceneDragDropEvent* event);
    virtual void dragMoveEvent(QGraphicsSceneDragDropEvent* event);
    virtual void dropEvent(QGraphicsSceneDragDropEvent* event);
};

#endif //CHESTNUT_SCENE_H