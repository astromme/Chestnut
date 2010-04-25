#include "datablock.h"

#include "ui_datablock.h"
#include "sizes.h"
#include "data.h"
#include "source.h"
#include "sink.h"

#include <QPainter>
#include <QDebug>
#include <QApplication>
#include <QFileDialog>

using namespace Chestnut;

DataBlock::DataBlock( const QString& name, const QString& datatype, int rows, int columns)
  : Data(name, Data::DataBlock, datatype)
{
  m_rows = rows;
  m_columns = columns;
  m_dimension = 2;
  m_ui = new Ui::DataBlockProperties;

  Sink *in = new Sink(Data::DataBlock, this);
  in->setPos(rect().left()+rect().width()/2, rect().top()-Size::inputHeight);
  m_sinks.append(in);
  
  Source *out = new Source(Data::DataBlock, this);
  out->setPos(rect().left()+rect().width()/2, rect().bottom());
  m_sources.append(out);
}

DataBlock::~DataBlock()
{

}

int DataBlock::type() const
{
  return Type;
}

int DataBlock::rows() const
{
  return m_rows;
}
int DataBlock::columns() const
{
  return m_columns;
}

ProgramStrings DataBlock::flatten() const
{
 
  if (isVisited()){
    return ProgramStrings();
  }
  setVisited(true);
  
  ProgramStrings prog;
  foreach(Sink *sink, sinks()){
    if (sink->isConnected()) {
      Data* sinkData = sink->sourceData();
      prog = prog + sinkData->flatten();
    }
  }
 
  QString datablockInChestnut = "vector";
  QString declaration;
  if (isInitialized()){
    declaration = datatype() + " " +
      name() + " " +
      QString::number(m_rows) + " " +
      QString::number(m_columns) + " " +
      expression() + ";";
  } else {
    declaration = datatype() + " " + name() + " " + datablockInChestnut + ";";
  }
  
  prog.first.append(declaration);
  
  foreach(Source *source, sources()){
    QList<Data*> sourceData = source->connectedData();
    foreach (Data* sData, sourceData){
      prog = prog + sData->flatten();
    }
  }
  
  return prog;
}

void DataBlock::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event)
{
  QDialog *dialog = new QDialog();
  m_ui->setupUi(dialog);
  
  m_ui->name->setValidator(m_nameValidator);
  m_ui->name->setText(name());
  
  if (datatype() == "int") {
    m_ui->integers->setChecked(true);
  } else {
    m_ui->realnumbers->setChecked(true);
  }
  m_ui->rows->setValue(m_rows);
  m_ui->columns->setValue(m_columns);
  
  //TODO for loop vs read from file
  connect(m_ui->fileSelect, SIGNAL(clicked(bool)), SLOT(selectFile()));
  connect(dialog, SIGNAL(accepted()), SLOT(configAccepted()));
  connect(dialog, SIGNAL(rejected()), SLOT(configRejected()));
  dialog->show();
  
  //FIXME: copied from file code below. Do this after show because we want widths
  int width = m_ui->selectedFile->width();  
  QString smallText = QApplication::fontMetrics().elidedText(m_inputFile, Qt::ElideLeft, width);
  m_ui->selectedFile->setText(smallText);
}

void DataBlock::configAccepted()
{
  setName(m_ui->name->text());
  if (m_ui->integers->isChecked()) {
    setDatatype("int");
  } else {
    setDatatype("float");
  }
  m_rows = m_ui->rows->value();
  m_columns = m_ui->columns->value();
  //TODO for loop vs read from file
  update();
}

void DataBlock::configRejected()
{
}

void DataBlock::selectFile()
{
  m_inputFile = QFileDialog::getOpenFileName(m_ui->fileSelect,
    tr("Select a Data File"), "~/", tr("Data Files (*.chestnutdata *.data)"));
    
  int width = m_ui->selectedFile->width();  
  QString smallText = QApplication::fontMetrics().elidedText(m_inputFile, Qt::ElideLeft, width);
  m_ui->selectedFile->setText(smallText);
}


QRectF DataBlock::rect() const
{
  return QRectF(QPointF(0, 0), QSizeF(Size::dataBlockWidth, Size::dataBlockHeight));
}

QRectF DataBlock::boundingRect() const
{
  return rect().adjusted(-1, -1, 1, 1);
}

void DataBlock::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
  painter->save();
  if (isSelected()) {
    painter->setPen(QPen(Qt::DashLine));
  }
  painter->drawRect(rect());
  painter->restore();
  
  painter->drawText(rect(), Qt::AlignBottom | Qt::AlignHCenter, name());
  painter->drawText(rect(), Qt::AlignTop | Qt::AlignHCenter, QString("%1 rows").arg(rows()));
  QRectF smaller = rect().adjusted(0, QApplication::fontMetrics().height(), 0, 0);
  painter->drawText(smaller, Qt::AlignTop | Qt::AlignHCenter, QString("%1 cols").arg(columns()));
}
