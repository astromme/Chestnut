
#include <QApplication>
#include <QWidget>
#include <QDebug>

#include "ui_mainwindow.h"

#include "value.h"
#include "function.h"
#include "source.h"
#include "sink.h"
#include "map.h"

int main(int argc, char* argv[]) {
  QApplication app(argc, argv);
  
  QMainWindow *widget = new QMainWindow();
  Ui::MainWindow ui;
  ui.setupUi(widget);

  widget->show();
  
  QGraphicsScene s;
  
  Value *v = new Value("var");
  v->setPos(0, -50);
  Map m;
  m.setPos(0, 50);
  Map m2;
  Map m3;
  
  m3.sources()[0]->connectToSink(m.sinks()[0]);
  v->sources()[0]->connectToSink(m.sinks()[1]);
  m.sources()[0]->connectToSink(m2.sinks()[0]);
  
  qDebug() << m.flatten();
 
  s.addItem(v);
  s.addItem(&m);
  
  ui.workflowEditor->setScene(&s);
  ui.workflowEditor->setRenderHint(QPainter::Antialiasing);
  ui.workflowEditor->setDragMode(QGraphicsView::RubberBandDrag);
  
  app.exec();
}