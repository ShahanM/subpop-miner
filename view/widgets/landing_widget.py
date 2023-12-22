from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, \
	QFileDialog, QProgressBar, QGridLayout
from utils.data_utils import DataContext
from utils.worker import Worker
import pandas as pd
from PyQt6.QtCore import pyqtSignal, QThreadPool, Qt

class LandingWidget(QWidget):
	def __init__(self, threadpool: QThreadPool, data_context: DataContext) \
		-> None:
		super(LandingWidget, self).__init__()
		
		self.threadpool = threadpool
		self.data_context = data_context
		
		# self.vbox = QVBoxLayout()	
		self.vbox = QGridLayout()
		self.vbox.setContentsMargins(0, 0, 0, 0)
		label1 = QLabel('Please use the open button below to choose a CSV file with headers.')
		label2 = QLabel('Note: Header names will be used as variable names in the rest of this application.')
		
		self.progress_placeholder = QVBoxLayout()

		file_dialog_button = QPushButton("Open File")
		file_dialog_button.pressed.connect(self.open_dialog)

		self.vbox.addWidget(label1, 0, 0, 9, 0, Qt.AlignmentFlag.AlignTop)
		self.vbox.addWidget(label2, 2, 0, 9, 0, Qt.AlignmentFlag.AlignTop)
		self.vbox.addWidget(file_dialog_button, 3, 0, 6, 0, Qt.AlignmentFlag.AlignCenter)
		self.vbox.addLayout(self.progress_placeholder, 5, 0, 9, 3, Qt.AlignmentFlag.AlignCenter)

		self.setLayout(self.vbox)

		self.progressbar = QProgressBar()
		self.progressbar.setValue(0)

	def open_dialog(self):
		fname = QFileDialog.getOpenFileName(
			self,
			"Open file",
			"${HOME}",
			"CSV Files (*.csv)"
		)
		print(fname)
		if fname[0]:
			self.load_csv_file(fname[0])

	def load_csv_file(self, filepath: str, sep: str = ',', \
		encoding: str = 'utf-8', chunksize: int = 100000):

		totalrows = sum(1 for row in open(filepath, 'r', encoding=encoding))
		self.progress_placeholder.addWidget(self.progressbar)
		print("Total rows: %d" % totalrows)
		worker = Worker(self.load_data_from_file, filepath, sep, encoding, \
			chunksize, totalrows)
		worker.signals.result.connect(self.on_file_load_complete)
		worker.signals.finished.connect(self.thread_complete)
		worker.signals.progress.connect(self.progress_fn)
		self.threadpool.start(worker)

	def load_data_from_file(self, filename: str,\
		sep: str, encoding: str, \
		chunksize: int, totalrows: int,\
		progress_callback) -> pd.DataFrame:
		data = []
		prog = 0
		progress_callback.emit(int(prog/totalrows*100))
		for chunk in pd.read_csv(filename, sep=sep, encoding=encoding, \
			chunksize=chunksize):
			data.append(chunk)
			prog += len(chunk)
			progress_callback.emit(int(prog/totalrows*100))
		
		return pd.concat(data)
		
	def on_file_load_complete(self, datafile: pd.DataFrame):
		self.progressbar.setValue(100)
		self.data_context.add_key_value_pair('dataframe', datafile)
		self.data_context.add_key_value_pair('variables', list(datafile.columns))

	def thread_complete(self):
		self.progress_placeholder.removeWidget(self.progressbar)

	def progress_fn(self, n):
		self.progressbar.setValue(n)
