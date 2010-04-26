#include "mainwindow.h"

#include <QGraphicsScene>
#include <QToolBar>
#include <QDebug>
#include <QTextEdit>
#include <QStandardItemModel>
#include <QStandardItem>

#include "ui_output.h"
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

