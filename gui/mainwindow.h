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

/**
 * This class is the main interface for the Chestnut GUI. It sets up
 * the window and provides the global user interactions, as well as support
 * for building and running Chestnut code.
 */
class MainWindow : public QMainWindow
{
  Q_OBJECT
  public:
    MainWindow(QWidget* parent = 0, Qt::WindowFlags flags = 0);
    virtual ~MainWindow();
    
  private slots:
    void clear();
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
