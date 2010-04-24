
#include <QApplication>
#include <QWidget>
#include <QDebug>
#include <QFile>

#include "ui_mainwindow.h"

#include "value.h"
#include "function.h"
#include "source.h"
#include "sink.h"
#include "map.h"
#include "datablock.h"

void writeToFile(QString fname, ProgramStrings prog);

int main(int argc, char* argv[]) {
  QApplication app(argc, argv);
  
  QMainWindow *widget = new QMainWindow();
  Ui::MainWindow ui;
  ui.setupUi(widget);

  widget->show();
  
  QGraphicsScene s;
  
  Value *v = new Value("var");
  Map *m1 = new Map();
//   Map *m2 = new Map();
//   Map *m3 = new Map();
  
  DataBlock *inmap = new DataBlock("inmap1", "float", 10, 10);
  DataBlock *outmap = new DataBlock("inmap2", "float", 10, 10);
  
  inmap->moveBy(-110, -150);
  outmap->moveBy(-50, 100);
  
//   m2->sources()[0]->connectToSink(m1->sinks()[0]);
  inmap->sources()[0]->connectToSink(m1->sinks()[0]);
  v->sources()[0]->connectToSink(m1->sinks()[1]);
  m1->sources()[0]->connectToSink(outmap->sinks()[0]);
//   m1->sources()[0]->connectToSink(m3->sinks()[0]);
  
  ProgramStrings prog = m1->flatten();
  writeToFile("DynamicChestnut.in", prog);
/*
  qDebug() << m2->flatten();
  qDebug() << m3->flatten();
*/
 
  s.addItem(v);
  s.addItem(m1);
//   s.addItem(m2);
//   s.addItem(m3);
  s.addItem(inmap);
  s.addItem(outmap);
  
  v->moveBy(0, -150);
  m1->moveBy(0, 0);
//   m2->moveBy(-50, -100);
//   m3->moveBy(-10, 100);
  
  ui.workflowEditor->setScene(&s);
  ui.workflowEditor->setRenderHint(QPainter::Antialiasing);
  ui.workflowEditor->setDragMode(QGraphicsView::RubberBandDrag);
 
  app.exec();
}

void writeToFile(QString fname, ProgramStrings prog)
{
  Declarations declarations = prog.first;
  Executions executions = prog.second;
  
  QFile file(fname);
  if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
      return;

  QTextStream out(&file);
  
  foreach (QString dec, declarations){
    out << dec << "\n";
  }
  
  out << "\n";
  
  foreach (QString exec, executions){
    out << exec << "\n";
  }
}

