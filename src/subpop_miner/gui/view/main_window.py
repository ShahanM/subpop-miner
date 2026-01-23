from utils.data_utils import DataContext
from view.layouts.file_loader_layout import FileLoaderLayout
from view.widgets.varpick_widget import VariablePickerWidget
from view.widgets.vartype_widget import VariableTypeIndicatorWidget
from view.widgets.mining_settings import MiningSettingsWidget
from view.widgets.console_widget import ConsoleWidget
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QWidget
from PyQt6.QtCore import QThreadPool, Qt


class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()

		self.data_context = DataContext()
		self.data_context.add_data({'dataframe': None, 'variables': [], 'accepted_variables': [], 'variable_types': {}})

		self.panel_index = 0

		self.setWindowTitle('Subpopulation Miner')

		self.threadpool = QThreadPool()
		print('Multithreading with maximum %d threads' % self.threadpool.maxThreadCount())

		# parent layout for application window
		self.main_layout = QVBoxLayout()

		# top running header for the main_layout
		# main_header_label = QLabel('Instruction...')

		# screen 1 - intro and data loader
		landing_widget = FileLoaderLayout(self.threadpool, self.data_context)

		# landing_widget = LandingWidget(self.threadpool, self.data_context)

		# screen 2 - variable picker
		self.var_pick_widget = VariablePickerWidget(self)

		# screen 3 - variable type indicator
		self.var_type_widget = VariableTypeIndicatorWidget(self.data_context, mainwindow=self)

		# screen 4 - parameters
		self.mining_params_widget = MiningSettingsWidget(self.data_context)

		# screen 5 - mining progress console
		mining_progress_widget = ConsoleWidget(self.threadpool, self.data_context)

		# screen 4 - report metadata (optional)
		rep_meta_widget = QWidget()

		# create the widget stack for the main panel
		self.widget_stack = QStackedWidget(self)
		self.widget_stack.addWidget(landing_widget)
		self.widget_stack.addWidget(self.var_pick_widget)
		self.widget_stack.addWidget(self.var_type_widget)
		self.widget_stack.addWidget(self.mining_params_widget)
		self.widget_stack.addWidget(mining_progress_widget)
		# self.widget_stack.addWidget(rep_meta_widget)
		self.widget_stack.currentChanged.connect(self.stack_changed)

		# footer navigation for main_layout
		hbox = QHBoxLayout()
		btn_back = QPushButton('Previous')
		btn_back.pressed.connect(self.go_previous_panel)
		hbox.addWidget(btn_back)

		btn_next = QPushButton('Next')
		btn_next.pressed.connect(self.go_next_panel)
		hbox.addWidget(btn_next)

		self.main_layout.addWidget(self.widget_stack, 0, Qt.AlignmentFlag.AlignCenter)
		self.main_layout.addLayout(hbox)

		main_widget = QWidget()
		main_widget.setLayout(self.main_layout)

		self.setCentralWidget(main_widget)
		self.show()

	def stack_changed(self, index):
		print('Stack changed to index: %d' % index)
		print(len(self.data_context.get_value('accepted_variables')) == 0)
		if index == 1 and self.var_pick_widget.left_list_widget.count() == 0:
			self.var_pick_widget.update_widget()
		if index == 2 and len(self.data_context.get_value('accepted_variables')) >= 0:
			self.var_type_widget.update_widget()
		# if index == 3:
		# self.mining_params_widget.update_widget()

	def go_next_panel(self):
		self.panel_index += 1
		self.widget_stack.setCurrentIndex(self.panel_index)

	def go_previous_panel(self):
		self.panel_index -= 1
		self.widget_stack.setCurrentIndex(self.panel_index)
