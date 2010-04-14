
#include <QApplication>
#include <QWidget>

#include "ui_mainwindow.h"

#include "value.h"
#include "function.h"

int main(int argc, char* argv[]) {
  QApplication app(argc, argv);
  
  QMainWindow *widget = new QMainWindow;
  Ui::MainWindow ui;
  ui.setupUi(widget);

  widget->show();
  
  QGraphicsScene s;
  Value *v = new Value("var");
  Function *f = new Function("function");
  s.addItem(v);
  s.addItem(f);
  
  f->moveBy(50, 50);
  
  ui.workflowEditor->setScene(&s);
  ui.workflowEditor->setRenderHint(QPainter::Antialiasing);
  
  app.exec();
}