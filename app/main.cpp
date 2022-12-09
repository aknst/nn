#include "mainwindow.h"

#include <QApplication>

MainWindow *wRef;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    wRef=&w;
    w.show();
    return a.exec();
}
