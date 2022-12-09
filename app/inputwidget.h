#ifndef INPUTWIDGET_H
#define INPUTWIDGET_H

#include <QWidget>
#include <QFrame>
#include <QImage>
#include <QPen>
#include <QBrush>
#include <vector>

class InputWidget : public QFrame
{
private:
    QImage buffImg;
    bool drawing;
    bool erasing;
    QPoint last;
    QImage image;

    QBrush myBrush;
    QPen myPen;
public:
    int penWidth;
    InputWidget(QWidget* parent);

public slots:
    void clear();
    void updatePenWidth(int value);
    std::vector<double> read();
    // QWidget interface
protected:
    void drawLine(QPoint last, QPoint now, QPainter& p, QColor c);
    void mousePressEvent(QMouseEvent *);
    void mouseReleaseEvent(QMouseEvent *);
    void mouseMoveEvent(QMouseEvent *);
    void paintEvent(QPaintEvent *);
    void resizeEvent(QResizeEvent *);

};

#endif // INPUTWIDGET_H
