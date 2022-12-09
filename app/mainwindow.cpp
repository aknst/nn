#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "nn.h"
#include <QFileDialog>

neural_net nn;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QString path = QCoreApplication::applicationDirPath() + "/model.txt";
    fstream fx(path.toStdString());
    if (fx.is_open()) {
        nn = neural_net(path.toStdString());
        ui->statusbar->showMessage("Модель model.txt загружена!");
    } else {
        ui->statusbar->showMessage("Модель сети не загружена!");
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}



void MainWindow::on_clearButton_clicked()
{
    ui->inputWidget->clear();
}

void MainWindow::update_labelPixmap(QImage& img) {
    ui->label->setPixmap(QPixmap::fromImage(img));
    ui->label->show();
}


void MainWindow::recognize()
{
    ui->probList->clear();
    auto res = ui->inputWidget->read();
    // qDebug() << res;
    if (!nn.ready) return;
    vvd current_input;
    current_input.push_back(res);
    auto z = nn.predict(current_input);
    int ans = argmax(z[0]);
    auto txt = QString::number(ans);
    for (int i = 0; i < 10; i++) {
        QString strValue = QString::number(z[0][i]*100, 'f', 3);
        auto t = QString::number(i) + " : ";
        t += strValue + "%";
        ui->probList->insertItem(i, t);
    }
    ui->answer->setText(txt);
}


void MainWindow::on_loadModelAction_triggered()
{
    QString filePath = QFileDialog::getOpenFileName(
                this,
                ("Загрузка модели"),
                "./",
                ("Text (*.txt)")
    );
    if (!filePath.isNull()) {
        nn = neural_net(filePath.toStdString());
        ui->statusbar->showMessage("Загружена модель: " + filePath);

    }
}

void MainWindow::on_horizontalSlider_valueChanged(int value)
{
    ui->inputWidget->updatePenWidth(value);
}

