#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QProcess>
#include <QElapsedTimer>
#include <QFileDialog>
#include <QMessageBox>
#include <QtCharts>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // Set up the table headers
    ui->tableWidgetCPU->setColumnCount(3);
    ui->tableWidgetCPU->setHorizontalHeaderLabels(QStringList() << "Run" << "Argument" << "Output");
    ui->tableWidgetCPU->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

    // Initialize chart
    chart = new QChart();
    chart->setTitle("Performance Graph");
    chart->legend()->setVisible(true);
    chart->legend()->setAlignment(Qt::AlignBottom);

    // Create series for the chart
    series = new QLineSeries();
    series->setName("Performance");

    // Add series to the chart
    chart->addSeries(series);

    // Create axes for the chart
    QValueAxis *axisX = new QValueAxis();
    axisX->setTitleText("Argument Value");
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);

    QValueAxis *axisY = new QValueAxis();
    axisY->setTitleText("Output Value");
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);

    // Create chart view
    chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    // Add chart view to the layout
    ui->chartLayout->addWidget(chartView);

    // Connect table data changed signal to update chart
    connect(ui->tableWidgetCPU, &QTableWidget::itemChanged, this, &MainWindow::updateChart);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_startBTN_clicked()
{
    // Clear previous results
    ui->tableWidgetCPU->setRowCount(0);

    // Clear previous chart data
    series->clear();

    // Open file dialog to select the program to run
    QString programPath = QFileDialog::getOpenFileName(this,
                                                       tr("Select Program to Run"),
                                                       QString(),
                                                       tr("All Files (*)"));

    if (programPath.isEmpty()) {
        return; // User canceled the file selection
    }

    // Get arguments from the line edit
    QString argumentsStr;
    if (ui->argsLineEdit) {
        argumentsStr = ui->argsLineEdit->text();
    }

    // Convert the input argument to a number
    bool ok;
    int fullNumber = argumentsStr.toInt(&ok);

    if (!ok) {
        QMessageBox::critical(this, tr("Error"), tr("Please enter a valid number as argument."));
        return;
    }

    // Disconnect the signal temporarily to avoid multiple updates
    disconnect(ui->tableWidgetCPU, &QTableWidget::itemChanged, this, &MainWindow::updateChart);

    // Data points for the chart
    QList<QPointF> points;

    // Run the process 10 times with incremental fractions of the number
    for (int i = 1; i <= 10; i++) {
        int currentValue = (fullNumber * i) / 10;
        QString currentArg = QString::number(currentValue);

        QProcess process;

        // Configure process to capture output
        process.setProcessChannelMode(QProcess::MergedChannels);

        // Start the process with the current argument
        QStringList argsList;
        argsList << currentArg;

        process.start(programPath, argsList);

        QString output;

        if (!process.waitForStarted()) {
            output = tr("Failed to start the program");
        } else {
            // Wait for the process to finish
            if (!process.waitForFinished(-1)) { // -1 means no timeout
                output = tr("Program execution failed");
            } else {
                // Get the output from the process
                output = QString::fromLocal8Bit(process.readAllStandardOutput());

                // Extract the last line (if there are multiple lines)
                if (!output.isEmpty()) {
                    QStringList lines = output.split('\n', Qt::SkipEmptyParts);
                    if (!lines.isEmpty()) {
                        output = lines.last();
                    }
                }
            }
        }

        // Add a new row to the table
        int row = ui->tableWidgetCPU->rowCount();
        ui->tableWidgetCPU->insertRow(row);

        // Add data to the row
        ui->tableWidgetCPU->setItem(row, 0, new QTableWidgetItem(QString("%1/10").arg(i)));
        ui->tableWidgetCPU->setItem(row, 1, new QTableWidgetItem(currentArg));
        ui->tableWidgetCPU->setItem(row, 2, new QTableWidgetItem(output));

        // Try to add point to the chart if output is a valid number
        bool outputOk;
        double outputValue = output.toDouble(&outputOk);
        if (outputOk) {
            qDebug() << "Adding point:" << currentValue << outputValue;
            points.append(QPointF(currentValue, outputValue));
        } else {
            qDebug() << "Could not convert output to number:" << output;
        }

        // Process events to update the UI
        QApplication::processEvents();
    }

    // Add all points to the series
    if (!points.isEmpty()) {
        series->replace(points);

        // Find min and max values for better axis scaling
        double minX = std::numeric_limits<double>::max();
        double maxX = std::numeric_limits<double>::lowest();
        double minY = std::numeric_limits<double>::max();
        double maxY = std::numeric_limits<double>::lowest();

        for (const QPointF &point : points) {
            minX = qMin(minX, point.x());
            maxX = qMax(maxX, point.x());
            minY = qMin(minY, point.y());
            maxY = qMax(maxY, point.y());
        }

        // Add some padding to the axes
        double xPadding = (maxX - minX) * 0.1;
        double yPadding = (maxY - minY) * 0.1;

        // Set axis ranges
        QValueAxis *axisX = qobject_cast<QValueAxis*>(chart->axes(Qt::Horizontal).first());
        QValueAxis *axisY = qobject_cast<QValueAxis*>(chart->axes(Qt::Vertical).first());

        if (axisX && axisY) {
            axisX->setRange(minX - xPadding, maxX + xPadding);
            axisY->setRange(minY - yPadding, maxY + yPadding);
        }

        qDebug() << "Chart updated with" << points.size() << "points";
    } else {
        qDebug() << "No valid data points to display";
    }

    // Reconnect the signal
    connect(ui->tableWidgetCPU, &QTableWidget::itemChanged, this, &MainWindow::updateChart);

    // Resize columns to content
    ui->tableWidgetCPU->resizeColumnsToContents();
}

void MainWindow::updateChart()
{
    // Clear existing data
    series->clear();

    // Collect data points
    QList<QPointF> points;

    // Get data from table and add to chart
    for (int row = 0; row < ui->tableWidgetCPU->rowCount(); ++row) {
        QTableWidgetItem* argItem = ui->tableWidgetCPU->item(row, 1);
        QTableWidgetItem* outputItem = ui->tableWidgetCPU->item(row, 2);

        if (argItem && outputItem) {
            bool argOk, outputOk;
            double argValue = argItem->text().toDouble(&argOk);
            double outputValue = outputItem->text().toDouble(&outputOk);

            if (argOk && outputOk) {
                points.append(QPointF(argValue, outputValue));
            }
        }
    }

    // Add all points to the series
    if (!points.isEmpty()) {
        series->replace(points);

        // Find min and max values for better axis scaling
        double minX = std::numeric_limits<double>::max();
        double maxX = std::numeric_limits<double>::lowest();
        double minY = std::numeric_limits<double>::max();
        double maxY = std::numeric_limits<double>::lowest();

        for (const QPointF &point : points) {
            minX = qMin(minX, point.x());
            maxX = qMax(maxX, point.x());
            minY = qMin(minY, point.y());
            maxY = qMax(maxY, point.y());
        }

        // Add some padding to the axes
        double xPadding = (maxX - minX) * 0.1;
        double yPadding = (maxY - minY) * 0.1;

        // Set axis ranges
        QValueAxis *axisX = qobject_cast<QValueAxis*>(chart->axes(Qt::Horizontal).first());
        QValueAxis *axisY = qobject_cast<QValueAxis*>(chart->axes(Qt::Vertical).first());

        if (axisX && axisY) {
            axisX->setRange(minX - xPadding, maxX + xPadding);
            axisY->setRange(minY - yPadding, maxY + yPadding);
        }
    }

    // Update the chart
    chartView->update();
}
