#include "inputwidget.h"
#include "mainwindow.h"
#include <QMouseEvent>
#include <QPainter>
#include <QDebug>
#include <vector>
#include <QProcess>

extern MainWindow *wRef;

InputWidget::InputWidget(QWidget* parent):
    QFrame(parent),image(280,280,QImage::Format_RGB888)
{
    image.fill(QColor::fromRgb(255,255,255));
    drawing = false;
    erasing = false;
    penWidth = 20;

    myBrush = QBrush(QColor::fromRgb(0,0,0));
    myPen = QPen(QColor::fromRgb(0,0,0));
    myPen.setCapStyle(Qt::RoundCap);
    myPen.setJoinStyle(Qt::RoundJoin);

    buffImg=image.copy();
}

void InputWidget::mousePressEvent(QMouseEvent *e){
    if(e->button() == Qt::LeftButton){
        last = e->pos();
        drawing = true;
    } else if (e->button() == Qt::RightButton) {
        last = e->pos();
        erasing = true;
    }
    e->accept();
}

void InputWidget::mouseMoveEvent(QMouseEvent *e){
    QPoint now = e->pos();
    QPainter p(&image);
    if (drawing){
        auto color = QColor::fromRgb(0,0,0);
        drawLine(last,now,p,color);
    } else if (erasing){
        auto color = QColor::fromRgb(255,255,255);
        drawLine(last,now,p,color);
    }
    update();
    wRef->recognize();
    last = now;
    e->accept();
}

void InputWidget::drawLine(QPoint last, QPoint now, QPainter& p, QColor c) {
    p.setBrush(myBrush);
    myPen.setWidth(penWidth);
    myPen.setColor(c);
    p.setPen(myPen);
    p.drawLine(last,now);
}

void InputWidget::mouseReleaseEvent(QMouseEvent *e){
    if (e->button() == Qt::LeftButton) {
        drawing = false;
    } else if (e->button() == Qt::RightButton) {
        erasing = false;
    }
    buffImg=image.copy();
    e->accept();
}

void InputWidget::paintEvent(QPaintEvent *e){
    QPainter p(this);
    QRect r = e->rect();
    p.drawImage(r,image);
    e->accept();
}

void InputWidget::resizeEvent(QResizeEvent *e){
    image = buffImg.scaled(e->size(),Qt::IgnoreAspectRatio,Qt::SmoothTransformation);
}


void InputWidget::clear(){
    image.fill(QColor::fromRgb(255,255,255));
    update();
}

void InputWidget::updatePenWidth(int value){
    penWidth = value;
}

std::vector<double> InputWidget::read(){
    int x1 = -1, y1 = -1, x2 = -1, y2 = -1;
    std::vector<double> result(784);
    for ( int y = 0; y < image.height(); y++ )
        for ( int x = 0; x < image.width(); x++ ) {
            QColor clrCurrent(image.pixel(x, y));
            int value = clrCurrent.value();
            if (value != 255) {
                if (y1 < 0) y1 = y;
                else y2 = y;
            }
        }
    for (int x = 0; x < image.width(); x++)
        for ( int y = 0; y < image.height(); y++) {
            QColor clrCurrent(image.pixel(x, y));
            int value = clrCurrent.value();
            if (value != 255) {
                if (x1 < 0) x1 = x;
                else x2 = x;
            }
        }

    QRect rect(x1, y1, x2-x1, y2-y1);
    QImage cropped = image.copy(rect);
    auto final = cropped.scaled(20,20,Qt::KeepAspectRatio, Qt::SmoothTransformation);

    QImage destImage;
    QImage *temp = new QImage(28,28,QImage::Format_RGB888);
    destImage = *temp;
    delete temp;


    destImage.fill(Qt::white);
    QPainter painter(&destImage);

    int deltaX = destImage.width() - final.width();
    int deltaY = destImage.height() - final.height();

    QPoint destPos = QPoint(deltaX/2, deltaY/2);
    painter.drawImage(destPos, final);
    painter.end();

    wRef->update_labelPixmap(destImage);

    int i = 0;
    for ( int y = 0; y < destImage.height(); ++y ) {
        for ( int x = 0; x < destImage.width(); ++x ) {
            QColor clrCurrent( destImage.pixel(x, y) );
            result[i] = (double)(255-clrCurrent.value())/255.0;
            i++;
        }
    }
    return result;
}
