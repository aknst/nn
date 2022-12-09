#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
// #include <nn.h>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
private slots:

    void on_clearButton_clicked();

    void on_loadModelAction_triggered();
    void on_horizontalSlider_valueChanged(int value);

public slots:
    void recognize();
    void update_labelPixmap(QImage &img);


private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
