from PyQt6.QtWidgets import (
	QWidget,
	QVBoxLayout,
	QLabel,
	QPushButton,
	QFileDialog,
	QProgressBar,
	QGridLayout,
	QListWidget,
)
from utils.data_utils import DataContext
from utils.worker import Worker
import pandas as pd
from PyQt6.QtCore import pyqtSignal, QThreadPool, Qt


class ConsoleWidget(QWidget):
	def __init__(self, threadpool: QThreadPool, data_context: DataContext) -> None:
		super(ConsoleWidget, self).__init__()

		self.threadpool = threadpool
		self.data_context = data_context

		# self.vbox = QVBoxLayout()
		self.vbox = QGridLayout()
		self.vbox.setContentsMargins(0, 0, 0, 0)
		label1 = QLabel('Please use the open button below to choose a CSV file with headers.')
		label2 = QLabel('Note: Header names will be used as variable names in the rest of this application.')

		self.console = QListWidget()
		self.progress_placeholder = QVBoxLayout()

		start_button = QPushButton('Start')
		start_button.pressed.connect(self.start)

		self.vbox.addWidget(label1, 0, 0, 9, 0, Qt.AlignmentFlag.AlignTop)
		self.vbox.addWidget(label2, 2, 0, 9, 0, Qt.AlignmentFlag.AlignTop)
		self.vbox.addWidget(self.console, 3, 0, 9, 9, Qt.AlignmentFlag.AlignTop)
		self.vbox.addLayout(self.progress_placeholder, 13, 0, 9, 9, Qt.AlignmentFlag.AlignTop)
		self.vbox.addWidget(start_button, 18, 0, 6, 0, Qt.AlignmentFlag.AlignCenter)

		self.setLayout(self.vbox)

		self.progressbar = QProgressBar()
		self.progressbar.setValue(0)
		self.progress_placeholder.addWidget(self.progressbar)

	def start(self):
		for i in range(100):
			self.console.addItem('Item %d' % i)
			# self.progressbar.setValue(i+1)

	def load_csv_file(self, filepath: str, sep: str = ',', encoding: str = 'utf-8', chunksize: int = 100000):
		totalrows = sum(1 for row in open(filepath, 'r', encoding=encoding))
		print('Total rows: %d' % totalrows)

		worker = Worker(self.run_mining, filepath, sep, encoding, chunksize, totalrows)

		worker.signals.result.connect(self.on_complete)
		worker.signals.finished.connect(self.thread_complete)
		worker.signals.progress.connect(self.progress_fn)
		self.threadpool.start(worker)

	def run_mining(self, data_context: DataContext, progress_callback) -> pd.DataFrame:
		data = []
		prog = 0
		progress_callback.emit(int(prog / totalrows * 100))
		for chunk in pd.read_csv(filename, sep=sep, encoding=encoding, chunksize=chunksize):
			data.append(chunk)
			prog += len(chunk)
			progress_callback.emit(int(prog / totalrows * 100))

		return pd.concat(data)

	def on_complete(self, datafile: pd.DataFrame):
		self.progressbar.setValue(100)
		# self.data_context.add_key_value_pair('dataframe', datafile)
		# self.data_context.add_key_value_pair('variables', list(datafile.columns))

	def thread_complete(self):
		# self.progress_placeholder.removeWidget(self.progressbar)
		print('THREAD COMPLETE!')

	def progress_fn(self, n):
		self.progressbar.setValue(n)
