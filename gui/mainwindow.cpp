#include "mainwindow.h"

#include <QGraphicsScene>
#include <QToolBar>
#include <QDebug>
#include <QTextEdit>
#include <QStandardItemModel>
#include <QStandardItem>

#include "ui_program.h"
#include "ui_mainwindow.h"

#include "scene.h"
#include "value.h"
#include "function.h"
#include "source.h"
#include "sink.h"
#include "map.h"
#include "reduce.h"
#include "sort.h"
#include "print.h"
#include "write.h"
#include "datablock.h"
#include "palettedelegate.h"
#include "palettemodel.h"

MainWindow::MainWindow(QWidget* parent, Qt::WindowFlags flags)
  : QMainWindow (parent, flags),
  m_ui(new Ui::MainWindow),
  m_outputUi(new Ui::OutputProgram)
{
  m_scene = new Scene(this);
  m_model = new PaletteModel(this);
  m_ui->setupUi(this);
  
  QStandardItem *variables = new QStandardItem("Variables");
    variables->appendRow(new QStandardItem("Data Block"));
    variables->appendRow(new QStandardItem("Value"));
    
  QStandardItem *functions = new QStandardItem("Functions");
    functions->appendRow(new QStandardItem("Map"));
    functions->appendRow(new QStandardItem("Reduce"));
    functions->appendRow(new QStandardItem("Sort"));
    functions->appendRow(new QStandardItem("Print"));
    functions->appendRow(new QStandardItem("Write"));
    
  QStandardItem *operators = new QStandardItem("Operators");
    operators->appendRow(new QStandardItem("Add"));
    operators->appendRow(new QStandardItem("Subtract"));
    operators->appendRow(new QStandardItem("Multiply"));
    operators->appendRow(new QStandardItem("Divide"));
  
  m_model->appendRow(variables);
  m_model->appendRow(functions);
  m_model->appendRow(operators);
   
  m_ui->palette->setModel(m_model);
  m_ui->palette->setItemDelegate(new PaletteDelegate);
  m_ui->palette->expandAll();
  
  connect(m_ui->actionBuild, SIGNAL(triggered(bool)), this, SLOT(writeFile()));
  
  // Create initial default objects
    
  Value *v1 = new Value("var");
  v1->setExpression("4");
  
  Value *v2 = new Value("reduced", "float");
  
  Map *m1 = new Map();
  Sort *s1 = new Sort();
  Write *w1 = new Write();
  //Print *p1 = new Print();
  //Reduce *r1 = new Reduce();
  
  w1->setFilename("chestnutOutput");
  
  DataBlock *inmap = new DataBlock("inmap1", "float", 10, 10);
  inmap->setExpression("foreach ( value = rand/RAND_MAX )");
  DataBlock *outmap = new DataBlock("outmap", "float", 10, 10);
  DataBlock *outsort = new DataBlock("sorted", "float", 10, 10);
  
  inmap->sources()[0]->connectToSink(m1->sinks()[0]);
  v1->sources()[0]->connectToSink(m1->sinks()[1]);
  m1->sources()[0]->connectToSink(outmap->sinks()[0]);
  
  outmap->sources()[0]->connectToSink(s1->sinks()[0]);
  s1->sources()[0]->connectToSink(outsort->sinks()[0]);
  
  outsort->sources()[0]->connectToSink(w1->sinks()[0]);
  
  //outmap->sources()[0]->connectToSink(r1->sinks()[0]);
  //r1->sources()[0]->connectToSink(v2->sinks()[0]);
  
  
  m_scene->addItem(v1);
  //m_scene->addItem(v2);

  m_scene->addItem(m1);
  m_scene->addItem(s1);
  m_scene->addItem(w1);
  //m_scene->addItem(p1);
  //m_scene->addItem(r1);

  m_scene->addItem(inmap);
  m_scene->addItem(outmap);
  m_scene->addItem(outsort);
  
  v1->moveBy(0, -150);
  //v2->moveBy(270, 180);
  
  m1->moveBy(0, 0);
  s1->moveBy(150,217);
  w1->moveBy(0,300);
  //r1->moveBy(150, 217);
  //p1->moveBy(0,300);
  
  inmap->moveBy(-110, -150);
  outmap->moveBy(-50, 100);
  outsort->moveBy(-270,180);
  
  m_ui->workflowEditor->setScene(m_scene);
  m_ui->workflowEditor->setAcceptDrops(true);
  m_ui->workflowEditor->setRenderHint(QPainter::Antialiasing);
  m_ui->workflowEditor->setDragMode(QGraphicsView::RubberBandDrag);
}
MainWindow::~MainWindow()
{
  delete m_ui;
  delete m_outputUi;
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
      QDialog *container = new QDialog();
      m_outputUi->setupUi(container);
      container->setWindowTitle("DynamicChestnut.in [Output] - Chestnut");
      
      m_outputUi->programCode->appendPlainText(prog.first.join("\n"));
      m_outputUi->programCode->appendPlainText(QString()); // new line
      m_outputUi->programCode->appendPlainText(prog.second.join("\n"));
      
      container->resize(450, 200);
      container->show();
      
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

