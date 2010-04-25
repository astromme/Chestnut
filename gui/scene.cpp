#include "scene.h"

#include <QGraphicsSceneMouseEvent>
#include <QDebug>
#include <QMimeData>


#include "value.h"
#include "datablock.h"

#include "map.h"
#include "reduce.h"
#include "sort.h"
#include "print.h"
#include "write.h"

#include "standardoperation.h"

Scene::Scene(QObject* parent)
  : QGraphicsScene(parent)
{
}

void Scene::dragEnterEvent(QGraphicsSceneDragDropEvent* event)
{
  qDebug() << "DragNDrop Enter";
  qDebug() << event->mimeData()->formats();
  if (event->mimeData()->hasFormat("application/x-chestnutpaletteitem")) {
    qDebug() << "accepting";
    event->setProposedAction(Qt::CopyAction);
    event->accept();
  }
}

void Scene::dragMoveEvent(QGraphicsSceneDragDropEvent* event)
{
  event->accept();
}

void Scene::dropEvent(QGraphicsSceneDragDropEvent* event)
{
  qDebug() << "DragNDrop Drop";
  if (event->mimeData()->hasFormat("application/x-chestnutpaletteitem")) {
    QByteArray encodedData = event->mimeData()->data("application/x-chestnutpaletteitem");
    QDataStream stream(&encodedData, QIODevice::ReadOnly);
    QStringList newItems;

    while (!stream.atEnd()) {
        QString text;
        stream >> text;
        newItems << text;
    }
    QString droppedItem = newItems.first();
    
    
    QGraphicsItem *newItem = 0;
    
    // Data
    if (droppedItem == "Value") { newItem = new Value("new value"); } 
    else if (droppedItem == "Data Block") { newItem = new DataBlock("new datablock", "float", 10, 10); }
    
    //Functions
    else if (droppedItem == "Map") { newItem = new Map(); }
    else if (droppedItem == "Reduce") { newItem = new Reduce(); }
    else if (droppedItem == "Sort") { newItem = new Sort(); }
    else if (droppedItem == "Print") { newItem = new Print(); }
    else if (droppedItem == "Write") { newItem = new Write(); }
    
    
    //Operators
    else if (droppedItem == "Plus") { newItem = new StandardOperation(StandardOperation::Add); }
  
    else {
      qDebug() << "Unknown Item: " << droppedItem;
      return;
    }
    
    addItem(newItem);
    newItem->setPos(event->scenePos());
    qDebug() << event->scenePos();
    return;
  }
  QGraphicsScene::dropEvent(event);
}
