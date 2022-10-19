import sys

from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from utils.data_context import DataContext
from widgets.landing_widget import LandingWidget
from widgets.varpick_widget import VariablePickerWidget
from widgets.vartype_widget import VariableTypeIndicatorWidget


class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()

		self.data_context = DataContext()
		self.data_context.add_data({
			'dataframe': None,
			'variables': [],
			'accepted_variables': []
		})

		self.panel_index = 0

		self.setWindowTitle("Subpopulation Miner")

		self.threadpool = QThreadPool()
		print("Multithreading with maximum %d threads" % \
			self.threadpool.maxThreadCount())

		# parent layout for application window
		self.main_layout = QVBoxLayout()

		# top running header for the main_layout
		main_header_label = QLabel('Instruction...')

		# screen 1 - intro and data loader
		landing_widget = LandingWidget(self.threadpool, self.data_context)

		# screen 2 - variable picker
		self.var_pick_widget = VariablePickerWidget(self.data_context)


		# screen 3 - variable type indicator
		self.var_type_widget = VariableTypeIndicatorWidget(self.data_context)

		# screen 4 - report metadata (optional)
		rep_meta_widget = QWidget()

		# create the widget stack for the main panel
		self.widget_stack = QStackedWidget(self)
		self.widget_stack.addWidget(landing_widget)
		self.widget_stack.addWidget(self.var_pick_widget)
		self.widget_stack.addWidget(self.var_type_widget)
		self.widget_stack.addWidget(rep_meta_widget)
		self.widget_stack.currentChanged.connect(self.stack_changed)

		# footer navigation for main_layout
		hbox = QHBoxLayout()
		main_back_button = QPushButton('Previous')
		main_back_button.pressed.connect(self.go_previous_panel)
		hbox.addWidget(main_back_button)

		main_next_button = QPushButton('Next')
		main_next_button.pressed.connect(self.go_next_panel)
		hbox.addWidget(main_next_button)

		self.main_layout.addWidget(main_header_label)
		self.main_layout.addWidget(self.widget_stack)
		self.main_layout.addLayout(hbox)
		
		main_widget = QWidget()
		main_widget.setLayout(self.main_layout)

		self.setCentralWidget(main_widget)
		self.show()
	
	def stack_changed(self, index):
		print('Stack changed to index: %d' % index)
		if index == 1:
			self.var_pick_widget.update_widget()
		if index == 2:
			self.var_type_widget.update_widget()

	def go_next_panel(self):
		self.panel_index += 1
		self.widget_stack.setCurrentIndex(self.panel_index)

	def go_previous_panel(self):
		self.panel_index -= 1
		self.widget_stack.setCurrentIndex(self.panel_index)	


if __name__ == '__main__':
	app = QApplication(sys.argv)
	w = MainWindow()
	app.exec()
