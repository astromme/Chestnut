
#include <QApplication>
#include <QWidget>
#include <QDebug>

#include "ui_mainwindow.h"

#include "value.h"
#include "function.h"
#include "source.h"
#include "sink.h"
#include "map.h"
#include "datablock.h"

int main(int argc, char* argv[]) {
  QApplication app(argc, argv);
  
  QMainWindow *widget = new QMainWindow();
  Ui::MainWindow ui;
  ui.setupUi(widget);

  widget->show();
  
  QGraphicsScene s;
  
  Value *v = new Value("var");
  Map *m1 = new Map();
  Map *m2 = new Map();
  Map *m3 = new Map();
  
  DataBlock *var = new DataBlock("var1", "float", 10, 10);
  s.addItem(var);
  var->moveBy(-50, 50);
  
  m2->sources()[0]->connectToSink(m1->sinks()[0]);
  v->sources()[0]->connectToSink(m1->sinks()[1]);
  m1->sources()[0]->connectToSink(m3->sinks()[0]);
  
  qDebug() << m1->flatten();
  qDebug() << m2->flatten();
  qDebug() << m3->flatten();
 
  s.addItem(v);
  s.addItem(m1);
  s.addItem(m2);
  s.addItem(m3);
  
  v->moveBy(0, -150);
  m1->moveBy(0, 0);
  m2->moveBy(-50, -100);
  m3->moveBy(-10, 100);
  
  ui.workflowEditor->setScene(&s);
  ui.workflowEditor->setRenderHint(QPainter::Antialiasing);
  ui.workflowEditor->setDragMode(QGraphicsView::RubberBandDrag);
 
  app.exec();
}
