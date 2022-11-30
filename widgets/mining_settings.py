from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, \
	QHBoxLayout, QComboBox, QRadioButton, QLineEdit, QGridLayout
from PyQt6.QtCore import Qt as qtcore


class MiningSettingsWidget(QWidget):
	def __init__(self, data_context) -> None:
		super(MiningSettingsWidget, self).__init__()

		self.data_context = data_context
		
		self.vbox = QGridLayout()
		self.vbox.setContentsMargins(0, 0, 0, 0)

		description_label = QLabel('Please select the parameters for the mining process. The numerical variable settings refer to the genetic algorithm settings. The categorical variable settings refer to the mining process.')
		description_label.setWordWrap(True)
		self.vbox.addWidget(description_label, 0, 0, 2, 9)

		self.setLayout(self.vbox)

	def update_widget(self):
		print('data_context accepted vars', self.data_context.accepted_variables)
		# FIXME - the list is duplicated when user clicks previous button and 
		# then next button
		self.__add_mining_settings()
		self.__add_genetic_settings()


	def __add_mining_settings(self):
		categorical_label = QLabel('Mining Categorical Variables Settings')
		categorical_label.setMaximumHeight(30)
		self.vbox.addWidget(categorical_label, 3, 0, 2, 5)

		minsup_label = QLabel('Minimum Support')
		self.vbox.addWidget(minsup_label, 4, 1, 1, 4, qtcore.AlignmentFlag.AlignLeft)

		mincat_var_label = QLabel('Maximum Categorical Variables')
		self.vbox.addWidget(mincat_var_label, 5, 1, 1, 4, qtcore.AlignmentFlag.AlignLeft)
		
		delta1_label = QLabel('Delta 1')
		self.vbox.addWidget(delta1_label, 6, 1, 1, 4, qtcore.AlignmentFlag.AlignLeft)

		delta2_label = QLabel('Delta 2')
		self.vbox.addWidget(delta2_label, 7, 1, 1, 4, qtcore.AlignmentFlag.AlignLeft)

		minsup_input = QLineEdit()
		minsup_input.setFixedWidth(144);
		self.vbox.addWidget(minsup_input, 4, 4)
		
		mincat_var_input = QLineEdit()
		mincat_var_input.setFixedWidth(144);
		self.vbox.addWidget(mincat_var_input, 5, 4)
		
		delta1_input = QLineEdit()
		delta1_input.setFixedWidth(144);
		self.vbox.addWidget(delta1_input, 6, 4)

		delta2_input = QLineEdit()
		delta2_input.setFixedWidth(144);
		self.vbox.addWidget(delta2_input, 7, 4)

	def __add_genetic_settings(self):
		numerical_label = QLabel('Genetic Algorithm Settings')
		numerical_label.setMaximumHeight(30)
		self.vbox.addWidget(numerical_label, 9, 0, 2, 5)

		population_size_label = QLabel('Population Size')
		self.vbox.addWidget(population_size_label, 10, 1, 1, 4, qtcore.AlignmentFlag.AlignLeft)

		max_generations_label = QLabel('Maximum Generations')
		self.vbox.addWidget(max_generations_label, 11, 1, 1, 4, qtcore.AlignmentFlag.AlignLeft)
		
		crossover_rate_label = QLabel('Crossover Rate')
		self.vbox.addWidget(crossover_rate_label, 12, 1, 1, 4, qtcore.AlignmentFlag.AlignLeft)

		mutation_rate_label = QLabel('Mutation Rate')
		self.vbox.addWidget(mutation_rate_label, 13, 1, 1, 4, qtcore.AlignmentFlag.AlignLeft)

		population_size_input = QLineEdit()
		population_size_input.setFixedWidth(144);
		self.vbox.addWidget(population_size_input, 10, 4)

		max_generations_input = QLineEdit()
		max_generations_input.setFixedWidth(144);
		self.vbox.addWidget(max_generations_input, 11, 4)

		crossover_rate_input = QLineEdit()
		crossover_rate_input.setFixedWidth(144);
		self.vbox.addWidget(crossover_rate_input, 12, 4)

		mutation_rate_input = QLineEdit()
		mutation_rate_input.setFixedWidth(144);
		self.vbox.addWidget(mutation_rate_input, 13, 4)
