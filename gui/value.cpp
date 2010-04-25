#include "value.h"

#include "sizes.h"
#include "drawingutils.h"
#include "source.h"
#include "sink.h"
#include "ui_value.h"

#include <QPainter>
#include <QApplication>
#include <QDebug>

using namespace Chestnut;

Value::Value( const QString& name, const QString& datatype)
  : Data(name, Data::Value, datatype)
{
  m_name = name;
  m_intValue = 0;
  m_floatValue = 0;
  m_ui = new Ui::ValueProperties;
  
  Sink *inputValue = new Sink(Data::Value, this);
  m_sinks.append(inputValue);
  inputValue->setPos(Size::valueWidth/2 - Size::inputWidth/2, 0);
  
  Source *outputValue = new Source(Data::Value, this);
  m_sources.append(outputValue);
  
  qreal xpos = Size::valueWidth/2 - Size::outputWidth/2;
  outputValue->setPos(QPointF(xpos, Size::valueHeight));
}

Value::~Value()
{

}

int Value::type() const
{
  return Type;
}

ProgramStrings Value::flatten() const
{
  if (isVisited()){
    return ProgramStrings();
  }
  setVisited(true);

  ProgramStrings prog;
  foreach(Sink *sink, sinks()){
    if (sink->isConnected()) {
      Data* sinkData = sink->sourceData();
      //ps += sinkData->flatten();
      prog = prog + sinkData->flatten();
    }
  }
 
  if (!isInitialized()){
    QString valueInChestnut = "scalar";
    QString declaration = datatype() + " " + name() + " " + valueInChestnut + ";";
    prog.first.append(declaration);
  }
  
  foreach(Source *source, sources()){
    QList<Data*> sourceData = source->connectedData();
    foreach (Data* sData, sourceData){
      prog = prog + sData->flatten();
    }
  }
  
  return prog;
}

void Value::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event)
{
  QDialog *dialog = new QDialog();
  m_ui->setupUi(dialog);
  
  m_ui->name->setValidator(m_nameValidator);
  m_ui->name->setText(name());
  
  if (datatype() == "int") {
    m_ui->integer->setChecked(true);
    m_ui->intValue->setValue(m_intValue);
    m_ui->intValue->setHidden(false);
    m_ui->realnumberValue->setHidden(true);
  } else {
    m_ui->realnumber->setChecked(true);
    m_ui->realnumberValue->setValue(m_floatValue);
    m_ui->realnumberValue->setHidden(false);
    m_ui->intValue->setHidden(true);
  }

  connect(dialog, SIGNAL(accepted()), SLOT(configAccepted()));
  connect(dialog, SIGNAL(rejected()), SLOT(configRejected()));
  dialog->show();
}

void Value::configAccepted()
{
  setName(m_ui->name->text());
  if (m_ui->integer->isChecked()) {
    setDatatype("int");
    m_intValue = m_ui->intValue->value();
  } else {
    setDatatype("float");
    m_floatValue = m_ui->realnumberValue->value();
  }

  update();
}
void Value::configRejected()
{

}


QRectF Value::boundingRect() const
{
  QPointF margin(1, 1);
  return QRectF(QPointF(0, 0) - margin,
                QPointF(Size::valueWidth, Size::valueHeight) + margin);
}

void Value::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
  
  QPointF center = QPointF(Size::valueWidth/2, Size::valueHeight/2);
  painter->save();
  if (isSelected()) {
    painter->setPen(QPen(Qt::DashLine));
  }
  painter->drawPath(triangle(center, Size::valueWidth, Size::valueHeight));
  painter->restore();
  
  // Layout and draw text
  qreal xpos = Size::valueWidth/2;
  qreal ypos = Size::valueHeight;
  
  xpos -= 0.5*QApplication::fontMetrics().width(m_name);
  ypos -= 5;
  painter->drawText(xpos, ypos, m_name);
  
  if (isInitialized()) {
    QString valueText;
    if (datatype() == "float") {
      valueText = QString("%L1").arg(m_floatValue, 0, 'f', 2);
    } else if (datatype() == "int") {
      valueText = QString::number(m_intValue);
    }
    painter->drawText(boundingRect(), Qt::AlignCenter, valueText);
  }
}
