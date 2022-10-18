from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import sys

import time
import pandas as pd

from landing_widget import LandingWidget
from data_context import DataContext
from vartype_widget import VariableTypeIndicatorWidget


class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()

		self.data_context = DataContext()
		self.data_context.add_data({
			'dataframe': None,
			'variables': [],
			'accepted_variables': []
		})

		self.selected_var_left = None
		self.selected_var_right = None

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
		# landing_widget = self.build_landing_widget()
		landing_widget = LandingWidget(self.threadpool, self.data_context)

		# screen 2 - variable picker
		var_pick_widget = self.build_var_pick_widget()

		# screen 3 - variable type indicator
		self.var_type_widget = VariableTypeIndicatorWidget(self.data_context)

		# screen 4 - report metadata (optional)
		rep_meta_widget = QWidget()

		# create the widget stack for the main panel
		self.widget_stack = QStackedWidget(self)
		self.widget_stack.addWidget(landing_widget)
		self.widget_stack.addWidget(var_pick_widget)
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
			self.left_list_widget.addItems(self.data_context.variables)
		if index == 2:
			self.var_type_widget.update_widget()

	def build_var_pick_widget(self):
		parent_widget = QWidget()

		hbox = QHBoxLayout()
		self.left_col = QVBoxLayout()
		self.mid_col = QVBoxLayout()
		self.right_col = QVBoxLayout()

		left_list_label = QLabel('Columns')
		self.left_col.addWidget(left_list_label)

		self.left_list_widget = QListWidget()
		self.left_list_widget.addItems(self.data_context.variables)
		self.left_list_widget.currentItemChanged.connect(self.list_item_changed)
		self.left_list_widget.currentTextChanged.connect(self.list_text_changed)
		self.left_col.addWidget(self.left_list_widget)

		move_item_right_button = QPushButton(">")
		move_item_right_button.pressed.connect(self.move_item_right)
		self.mid_col.addWidget(move_item_right_button)

		move_item_left_button = QPushButton("<")
		move_item_left_button.pressed.connect(self.move_item_left)
		self.mid_col.addWidget(move_item_left_button)

		right_list_label = QLabel('Selected')
		self.right_col.addWidget(right_list_label)
		self.right_list_widget = QListWidget()
		self.right_list_widget.addItems(self.data_context.accepted_variables)
		self.right_list_widget.currentItemChanged.connect(self.list_item_changed)
		self.right_list_widget.currentTextChanged.connect(self.list_text_changed)
		self.right_col.addWidget(self.right_list_widget)

		hbox.addLayout(self.left_col)
		hbox.addLayout(self.mid_col)
		hbox.addLayout(self.right_col)

		parent_widget.setLayout(hbox)

		return parent_widget

	def go_next_panel(self):
		self.panel_index += 1
		self.widget_stack.setCurrentIndex(self.panel_index)

	def go_previous_panel(self):
		self.panel_index -= 1
		self.widget_stack.setCurrentIndex(self.panel_index)
	
	def list_item_changed(self, item):
		if self.left_list_widget == item.listWidget():
			self.selected_var_left = item
		
		if self.right_list_widget == item.listWidget():
			self.selected_var_right = item
	
	def list_text_changed(self, text):
		print('text', text)

	def move_item_right(self):
		print('move right')
		if self.selected_var_left:
			self.right_list_widget.addItem(self.selected_var_left.text())
			self.data_context.accepted_variables.append(self.selected_var_left.text())
			self.left_list_widget.takeItem(self.left_list_widget.row(self.selected_var_left))
			self.data_context.variables.remove(self.selected_var_left.text())
			self.selected_var_left = None

	def move_item_left(self):
		print('move left')
		if self.selected_var_right:
			self.left_list_widget.addItem(self.selected_var_right.text())
			self.data_context.variables.append(self.selected_var_right.text())
			self.right_list_widget.takeItem(self.right_list_widget.row(self.selected_var_right))
			self.data_context.accepted_variables.remove(self.selected_var_right.text())
			self.selected_var_right = None
	

app = QApplication(sys.argv)
w = MainWindow()
app.exec()
