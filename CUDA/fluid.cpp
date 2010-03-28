#include "FluidView.h"

#include <QApplication>

int main(int argc, char* argv[]) {
  QApplication app(argc, argv);
  FluidView v(500, 500);
  v.show();
  app.exec();
  return 0;
}