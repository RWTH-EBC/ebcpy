"""Module with GUI for showing and altering tuner-parameters"""

from PyQt5 import QtCore, QtGui, QtWidgets
from ebcpy import data_types
import numpy as np
# pylint: disable=R0902

class TunerParasUI:
    """
    Class for the PyQt5-Window to show and
    alter given tuner-parameters
    """

    tuner_paras = None
    popmenu = None
    action = None

    def __init__(self, main_window):
        """Instantiate instance parameters, normally setupUI"""
        main_window.setObjectName("MainWindow")
        main_window.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(main_window)
        self.centralwidget.setObjectName("centralwidget")
        self.grid_layout = QtWidgets.QGridLayout(self.centralwidget)
        self.grid_layout.setObjectName("gridLayout")
        self.push_button_store = QtWidgets.QPushButton(self.centralwidget)
        self.push_button_store.setObjectName("push_button_store")
        self.grid_layout.addWidget(self.push_button_store, 1, 0, 1, 2)
        self.push_button_close = QtWidgets.QPushButton(self.centralwidget)
        self.push_button_close.setObjectName("push_button_close")
        self.grid_layout.addWidget(self.push_button_close, 1, 2, 1, 1)
        self.table_widget = QtWidgets.QTableWidget(self.centralwidget)
        self.table_widget.setObjectName("table_widget")
        self.table_widget.setColumnCount(4)
        self.table_widget.setSortingEnabled(True)
        self.table_widget.setRowCount(0)
        self.table_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table_widget.customContextMenuRequested.connect(self.right_click_menu)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(4, item)
        self.grid_layout.addWidget(self.table_widget, 0, 0, 1, 3)
        main_window.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(main_window)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        main_window.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(main_window)
        self.statusbar.setObjectName("statusbar")
        main_window.setStatusBar(self.statusbar)

        self.retranslate_ui(main_window)
        QtCore.QMetaObject.connectSlotsByName(main_window)

        self.set_connections()
        self.main_window = main_window

    def retranslate_ui(self, main_window):
        """Retranslate based on pyuic-converter"""
        _translate = QtCore.QCoreApplication.translate
        main_window.setWindowTitle(_translate("MainWindow", "Tuner Parameters"))
        self.push_button_store.setText(_translate("MainWindow", "Store to object"))
        self.push_button_close.setText(_translate("MainWindow", "Close"))
        item = self.table_widget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Name"))
        item = self.table_widget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Initial Value"))
        item = self.table_widget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Min"))
        item = self.table_widget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Max"))

    def right_click_menu(self):
        """Setup rigtht-click-event to delete some rows."""
        self.popmenu = QtWidgets.QMenu()
        # Fill the Menu with Functions
        self.action = self.popmenu.addAction("Delete Tuner Parameter")
        self.action.triggered.connect(self._delete_current_row)
        self.popmenu.move(QtGui.QCursor.pos())
        self.popmenu.show()

    def _delete_current_row(self):
        """Delete current row"""
        self.table_widget.removeRow(self.table_widget.currentRow())

    def set_connections(self):
        """Connect buttons"""
        self.push_button_store.clicked.connect(self.store_and_close)
        self.push_button_close.clicked.connect(self.close_app)

    def set_data(self, df):
        """Print the given datafranme to the tableWidget for editing.
        :param df: pd.DataFrame
            DataFrame to be loaded.
        """
        df["names"] = df.index
        df_rows, _ = df.shape
        self.table_widget.setRowCount(df_rows)
        # Set Index:
        for row in range(df_rows):
            for col, col_name in enumerate(["names", "initial_value", "min", "max"]):
                item = QtWidgets.QTableWidgetItem()
                value = df[col_name].iloc[row]
                if isinstance(value, str):
                    item.setFlags(QtCore.Qt.ItemIsEnabled)
                    item.setData(2, value)
                else:
                    if abs(value) == np.inf:
                        value = 0
                    if abs(value) > 2147483647:  # Maximal value-size
                        value = 2147483647

                    item.setData(2, float(value))
                self.table_widget.setItem(row, col, item)
        self.table_widget.resizeColumnsToContents()

    def store_and_close(self):
        """Try to generate and thus save the tuner_paras in the widget and later
        close the window."""
        names, initial_values, bounds = [], [], []
        for row in range(self.table_widget.rowCount()):
            names.append(self.table_widget.item(row, 0).text())
            initial_values.append(self.table_widget.item(row, 1).data(2))
            bounds.append((self.table_widget.item(row, 2).data(2),
                           self.table_widget.item(row, 3).data(2)))
        try:
            self.tuner_paras = data_types.TunerParas(names, initial_values, bounds)
            self.main_window.close()
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self.centralwidget, "Error", str(error))

    def close_app(self):
        """Close the application and set tuner_paras to None"""
        self.tuner_paras = None
        self.main_window.close()
