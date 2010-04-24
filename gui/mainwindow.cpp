#include "mainwindow.h"

#include <QGraphicsScene>
#include <QToolBar>
#include <QDebug>
#include <QTextEdit>

#include "value.h"
#include "function.h"
#include "source.h"
#include "sink.h"
#include "map.h"
#include "reduce.h"
#include "sort.h"
#include "print.h"
#include "datablock.h"

MainWindow::MainWindow(QWidget* parent, Qt::WindowFlags flags)
  : QMainWindow (parent, flags),
  m_ui(new Ui::MainWindow)
{
  m_scene = new QGraphicsScene(this);
  m_ui->setupUi(this);
  QToolBar *toolbar = addToolBar("Project Actions");
  toolbar->addAction(m_ui->actionBuild);
  
  connect(m_ui->actionBuild, SIGNAL(triggered(bool)), this, SLOT(writeFile()));
  
  // Create initial default objects
    
  Value *v1 = new Value("var");
  v1->setExpression("4");
  
  Value *v2 = new Value("reduced", "float");
  
  Map *m1 = new Map();
//   Map *m2 = new Map();
//   Map *m3 = new Map();

  //Reduce *r1 = new Reduce();
  Sort *s1 = new Sort();
  
  Print *p1 = new Print();
  
  DataBlock *inmap = new DataBlock("inmap1", "float", 10, 10);
  inmap->setExpression("foreach ( value = rand/RAND_MAX )");
  DataBlock *outmap = new DataBlock("outmap", "float", 10, 10);
  DataBlock *outsort = new DataBlock("sorted", "float", 10, 10);
  
//   m2->sources()[0]->connectToSink(m1->sinks()[0]);
  inmap->sources()[0]->connectToSink(m1->sinks()[0]);
  v1->sources()[0]->connectToSink(m1->sinks()[1]);
  m1->sources()[0]->connectToSink(outmap->sinks()[0]);
  
  outmap->sources()[0]->connectToSink(s1->sinks()[0]);
  s1->sources()[0]->connectToSink(outsort->sinks()[0]);
  
  //outsort->sources()[0]->connectToSink(p1->sinks()[0]);
    
  
  //outmap->sources()[0]->connectToSink(r1->sinks()[0]);
  //r1->sources()[0]->connectToSink(v2->sinks()[0]);
  
  
//   m1->sources()[0]->connectToSink(m3->sinks()[0]);
  
/*
  qDebug() << m2->flatten();
  qDebug() << m3->flatten();
*/
 
  m_scene->addItem(v1);
  //m_scene->addItem(v2);
  m_scene->addItem(m1);
  m_scene->addItem(s1);
  m_scene->addItem(p1);
  //m_scene->addItem(r1);
//   m_scene->addItem(m2);
//   m_scene->addItem(m3);
  m_scene->addItem(inmap);
  m_scene->addItem(outmap);
  m_scene->addItem(outsort);
  
  v1->moveBy(0, -150);
  //v2->moveBy(270, 180);
  
  m1->moveBy(0, 0);
  //r1->moveBy(150, 217);
  s1->moveBy(150,217);
  p1->moveBy(0,300);
  
  inmap->moveBy(-110, -150);
  outmap->moveBy(-50, 100);
  outsort->moveBy(-270,180);
  
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
  //TODO: Make less hacky
  foreach(QGraphicsItem *item, m_scene->items()) {
    if (item->type() != ChestnutItemType::Map) {
      continue;
    }
    Object *object = (Object*)item;
    if (object) {
      ProgramStrings prog = object->flatten();
      
      // Show resulting program in a window
      QTextEdit *resultingProgram = new QTextEdit();
      resultingProgram->append(prog.first.join("\n"));
      resultingProgram->append(QString()); // new line
      
      resultingProgram->append(prog.second.join("\n"));
      resultingProgram->resize(400, 200);
      resultingProgram->show();
      
      // Write resulting program to a file
      writeToFile("DynamicChestnut.in", prog);
      
      // Reset nodes so we can run this again
      unvisitAll();
      return;
    }
  }
}

void MainWindow::unvisitAll()
{
  foreach(QGraphicsItem *item, m_scene->items()) {
    Object *ob = qgraphicsitem_cast<Object*>(item);
    if (ob) {
      ob->setVisited(false);
    }
  }
}

