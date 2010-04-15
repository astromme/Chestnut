
#include <QApplication>
#include <QWidget>

#include "ui_mainwindow.h"

#include "value.h"
#include "function.h"
#include "source.h"
#include "sink.h"
#include "map.h"

int main(int argc, char* argv[]) {
  QApplication app(argc, argv);
  
  QMainWindow *widget = new QMainWindow;
  Ui::MainWindow ui;
  ui.setupUi(widget);

  widget->show();
  
  QGraphicsScene s;
  Source *source = new Source(Data::Value, 0);
  source->setPos(0, -25);
  Value *v = new Value("var");
  v->setPos(0, -50);

  s.addItem(v);
  s.addItem(source);
 
  Map m;
  m.setPos(0, 50);
  s.addItem(&m);
  //f->moveBy(50, 50);
  
  ui.workflowEditor->setScene(&s);
  ui.workflowEditor->setRenderHint(QPainter::Antialiasing);
  ui.workflowEditor->setDragMode(QGraphicsView::RubberBandDrag);
  
  app.exec();
}