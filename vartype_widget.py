from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, \
	QHBoxLayout, QComboBox

class VariableTypeIndicatorWidget(QWidget):
	def __init__(self, data_context) -> None:
		super(VariableTypeIndicatorWidget, self).__init__()

		self.data_context = data_context
		
		self.vbox = QVBoxLayout()
		label = QLabel('Variable Type')
		self.vbox.addWidget(label)

		self.setLayout(self.vbox)

	def update_widget(self):
		print('data_context accepted vars', self.data_context.accepted_variables)
		# FIXME - the list is duplicated when user clicks previous button and 
		# then next button
		header = QHBoxLayout()
		header.addWidget(QLabel('Selected Variables'))
		header.addWidget(QLabel('Data Type'))
		header.addWidget(QLabel('Side'))
		self.vbox.addLayout(header)

		for var in self.data_context.accepted_variables:
			row = QHBoxLayout()
			row.addWidget(QLabel(var))
			type_select = QComboBox()
			type_select.addItems(['Categorical', 'Numerical'])
			row.addWidget(type_select)

			kind_select = QComboBox()
			kind_select.addItems(['Dependent', 'Independent'])
			row.addWidget(kind_select)

			self.vbox.addLayout(row)