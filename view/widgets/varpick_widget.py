from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, \
	QListWidget, QPushButton


class VariablePickerWidget(QWidget):
	def __init__(self, mainwindow) -> None:
		super(VariablePickerWidget, self).__init__()
		self.mainwindow = mainwindow

		self.selected_var_left = None
		self.selected_var_right = None

		self.vbox = QVBoxLayout()
		self.desc = QLabel('Select variables to be used in the analysis which includes variables subject to protection and variables which define subpopulations.\n')
		self.desc.setWordWrap(True)

		hbox = QHBoxLayout()
		self.left_col = QVBoxLayout()
		self.mid_col = QVBoxLayout()
		self.right_col = QVBoxLayout()

		self.left_col = QVBoxLayout()
		self.mid_col = QVBoxLayout()
		self.right_col = QVBoxLayout()

		left_list_label = QLabel('Available Variables')
		self.left_col.addWidget(left_list_label)

		self.left_list_widget = QListWidget()
		self.left_list_widget.addItems(self.mainwindow.data_context.variables)
		self.left_list_widget.currentItemChanged.connect(self.list_item_changed)
		self.left_list_widget.currentTextChanged.connect(self.list_text_changed)
		self.left_col.addWidget(self.left_list_widget)

		move_item_right_button = QPushButton(">")
		move_item_right_button.pressed.connect(self.move_item_right)
		self.mid_col.addWidget(move_item_right_button)

		move_item_left_button = QPushButton("<")
		move_item_left_button.pressed.connect(self.move_item_left)
		self.mid_col.addWidget(move_item_left_button)

		right_list_label = QLabel('Variables Selected for Analysis')
		self.right_col.addWidget(right_list_label)
		self.right_list_widget = QListWidget()
		self.right_list_widget.addItems(self.mainwindow.data_context.accepted_variables)
		self.right_list_widget.currentItemChanged.connect(self.list_item_changed)
		self.right_list_widget.currentTextChanged.connect(self.list_text_changed)
		self.right_col.addWidget(self.right_list_widget)

		hbox.addLayout(self.left_col)
		hbox.addLayout(self.mid_col)
		hbox.addLayout(self.right_col)

		self.vbox.addWidget(self.desc)
		self.vbox.addLayout(hbox)
		self.setLayout(self.vbox)
		self.update_widget()


	def list_item_changed(self, item):
		if item:
			if self.left_list_widget == item.listWidget():
				self.selected_var_left = item
			
			if self.right_list_widget == item.listWidget():
				self.selected_var_right = item
	
	def list_text_changed(self, text):
		print('text', text)

	def move_item_right(self):
		if self.selected_var_left:
			self.right_list_widget.addItem(self.selected_var_left.text())
			self.mainwindow.data_context.accepted_variables.append(self.selected_var_left.text())
			self.left_list_widget.takeItem(self.left_list_widget.row(self.selected_var_left))
			if self.selected_var_left.text() in self.mainwindow.data_context.variables:
				self.mainwindow.data_context.variables.remove(self.selected_var_left.text())
			if self.left_list_widget.count() > 0:
				self.selected_var_left = self.left_list_widget.item(0)
			else:
				self.selected_var_left = None

	def move_item_left(self):
		if self.selected_var_right:
			self.left_list_widget.addItem(self.selected_var_right.text())
			self.mainwindow.data_context.variables.append(self.selected_var_right.text())
			self.right_list_widget.takeItem(self.right_list_widget.row(self.selected_var_right))
			if self.selected_var_right.text() in self.mainwindow.data_context.accepted_variables:
				self.mainwindow.data_context.accepted_variables.remove(self.selected_var_right.text())
			if self.right_list_widget.count() > 0:
				self.selected_var_right = self.right_list_widget.item(0)
			else:
				self.selected_var_right = None

	def update_widget(self):
		self.left_list_widget.addItems(self.mainwindow.data_context.variables)