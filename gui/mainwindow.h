#ifndef CHESTNUT_MAINWINDOW_H
#define CHESTNUT_MAINWINDOW_H

#include <QtGui/QMainWindow>

namespace Ui {
  class MainWindow;
  class OutputProgram;
  class RunOutput;
}

class QStandardItemModel;
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
    
    void runCompiledCode();
    
  private:
    QGraphicsScene *m_scene;
    QStandardItemModel *m_model;
    Ui::MainWindow *m_ui;
    Ui::OutputProgram *m_outputUi;
    Ui::RunOutput *m_runOutputUi;
};

#endif // CHESTNUT_MAINWINDOW_H
