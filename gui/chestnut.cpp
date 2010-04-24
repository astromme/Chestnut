
#include <QApplication>
#include <QWidget>
#include <QDebug>
#include <QFile>

#include "mainwindow.h"

int main(int argc, char* argv[]) {
  QApplication app(argc, argv);
  MainWindow mainWindow;

  mainWindow.show();
  
  app.exec();
}

