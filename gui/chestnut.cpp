
#include <QApplication>
#include <QWidget>

#include "ui_mainwindow.h"

#include "value.h"
#include "function.h"
#include "source.h"
#include "sink.h"

int main(int argc, char* argv[]) {
  QApplication app(argc, argv);
  
  QMainWindow *widget = new QMainWindow;
  Ui::MainWindow ui;
  ui.setupUi(widget);

  widget->show();
  
  QGraphicsScene s;
  Source *source = new Source(Data::Value, 0);
  Data::Types both;
  both << Data::Value;
  both << Data::DataBlock;
  Sink *sink = new Sink(both, 0);
  Value *v = new Value("var");
  v->setFlag(QGraphicsItem::ItemIsMovable);
  v->setFlag(QGraphicsItem::ItemIsFocusable);
  //Function *f = new Function("function");
  //s.addItem(v);
  s.addItem(source);
  s.addItem(sink);
  source->connectToSink(sink);
  sink->moveBy(50, 50);
  //s.addItem(f);
 
  
  //f->moveBy(50, 50);
  
  ui.workflowEditor->setScene(&s);
  ui.workflowEditor->setRenderHint(QPainter::Antialiasing);
  ui.workflowEditor->setDragMode(QGraphicsView::RubberBandDrag);
  
  app.exec();
}