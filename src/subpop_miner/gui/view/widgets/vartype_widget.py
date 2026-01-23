from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QComboBox, QRadioButton, QGridLayout
from PyQt6.QtCore import Qt as qtcore


class VariableTypeIndicatorWidget(QWidget):
	def __init__(self, data_context, mainwindow) -> None:
		super(VariableTypeIndicatorWidget, self).__init__()

		self.data_context = data_context

		# self.vbox = QVBoxLayout()
		self.vbox = QGridLayout()
		# label = QLabel('Variable Type')
		# self.vbox.addWidget(label)

		self.setLayout(self.vbox)
		self.update_widget()

	def update_widget(self):
		print('data_context accepted vars', self.data_context.accepted_variables)

		self.vbox.addWidget(
			QLabel('Select variable that is subject to extreme value protection:'),
			1,
			1,
			1,
			5,
			qtcore.AlignmentFlag.AlignHCenter | qtcore.AlignmentFlag.AlignTop,
		)
		var_select = QComboBox()
		var_select.addItems(self.data_context.accepted_variables)
		self.vbox.addWidget(var_select, 1, 6, 1, 4, qtcore.AlignmentFlag.AlignHCenter | qtcore.AlignmentFlag.AlignTop)
		# header = QHBoxLayout()

		desc = QLabel('Indicate the relevant data type for each variable.\n')
		desc.setWordWrap(True)
		self.vbox.addWidget(desc, 3, 1, 1, 9, qtcore.AlignmentFlag.AlignTop)

		self.vbox.addWidget(
			QLabel('Selected Variables'), 6, 1, 1, 3, qtcore.AlignmentFlag.AlignLeft | qtcore.AlignmentFlag.AlignTop
		)
		self.vbox.addWidget(
			QLabel('Data Type'), 6, 4, 1, 3, qtcore.AlignmentFlag.AlignHCenter | qtcore.AlignmentFlag.AlignTop
		)
		# self.vbox.addWidget(QLabel('Target'), 3, 7, 1, 2, qtcore.AlignmentFlag.AlignHCenter|qtcore.AlignmentFlag.AlignTop)
		# self.vbox.addLayout(header)

		max_ = 0
		for i, var in enumerate(self.data_context.accepted_variables, 7):
			# row = QHBoxLayout()
			self.vbox.addWidget(QLabel(var), i, 1, 1, 3, qtcore.AlignmentFlag.AlignLeft | qtcore.AlignmentFlag.AlignTop)
			type_select = QComboBox()
			type_select.addItems(['Categorical', 'Numerical'])
			type_select.currentTextChanged.connect(self.__picked_type)
			self.vbox.addWidget(
				type_select, i, 4, 1, 3, qtcore.AlignmentFlag.AlignHCenter | qtcore.AlignmentFlag.AlignTop
			)
			self.data_context.variable_types[var] = type_select.currentText()

			# target_select = QRadioButton()
			# self.vbox.addWidget(target_select, i, 7, 1, 2, qtcore.AlignmentFlag.AlignHCenter|qtcore.AlignmentFlag.AlignTop)

			# self.vbox.addLayout(row, i, 1)
			# max_ = i

	def __picked_type(self, text):
		# self.data_context.variable_types[var] = text
		# print(self.data_context.variable_types)
		print(text)
