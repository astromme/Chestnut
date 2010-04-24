#ifndef CHESTNUT_MAINWINDOW_H
#define CHESTNUT_MAINWINDOW_H

#include <QtGui/QMainWindow>
#include "ui_mainwindow.h"

class QGraphicsScene;

class MainWindow : public QMainWindow
{
  Q_OBJECT
  public:
    MainWindow(QWidget* parent = 0, Qt::WindowFlags flags = 0);
    virtual ~MainWindow();
    
  private slots:
    void writeFile();
    void unvisitAll();
    
  private:
    QGraphicsScene *m_scene;
    Ui::MainWindow *m_ui;
};

#endif // CHESTNUT_MAINWINDOW_H
