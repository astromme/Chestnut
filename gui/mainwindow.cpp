#include "mainwindow.h"

#include <QGraphicsScene>

#include "value.h"
#include "function.h"
#include "source.h"
#include "sink.h"
#include "map.h"
#include "datablock.h"

MainWindow::MainWindow(QWidget* parent, Qt::WindowFlags flags)
  : QMainWindow (parent, flags),
  m_ui(new Ui::MainWindow)
{
  m_scene = new QGraphicsScene(this);
  m_ui->setupUi(this);
  // Create initial default objects
    
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
 
  m_scene->addItem(v);
  m_scene->addItem(m1);
//   m_scene->addItem(m2);
//   m_scene->addItem(m3);
  m_scene->addItem(inmap);
  m_scene->addItem(outmap);
  
  v->moveBy(0, -150);
  m1->moveBy(0, 0);
//   m2->moveBy(-50, -100);
//   m3->moveBy(-10, 100);
  
  m_ui->workflowEditor->setScene(m_scene);
  m_ui->workflowEditor->setRenderHint(QPainter::Antialiasing);
  m_ui->workflowEditor->setDragMode(QGraphicsView::RubberBandDrag);
}
MainWindow::~MainWindow()
{

}


void MainWindow::writeFile()
{
  
}
