
#include <QApplication>
#include <QWidget>

#include "ui_mainwindow.h"

#include "value.h"

int main(int argc, char* argv[]) {
  QApplication app(argc, argv);
  
  QMainWindow *widget = new QMainWindow;
  Ui::MainWindow ui;
  ui.setupUi(widget);

  widget->show();
  
  QGraphicsScene s;
  Value *v = new Value();
  s.addItem(v);
  
  ui.workflowEditor->setScene(&s);
  
  app.exec();
}