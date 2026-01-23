from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QProgressBar, QGridLayout
from utils.data_utils import DataContext
from utils.worker import Worker
import pandas as pd
from PyQt6.QtCore import pyqtSignal, QThreadPool, Qt

from res.strings import DEFAULT_STR as str_local
from view.styles.stylesheet import get_default_margin


class FileLoaderLayout(QWidget):
	def __init__(self, threadpool: QThreadPool, data_context: DataContext) -> None:
		super(FileLoaderLayout, self).__init__()

		# Load strings for the layout
		self.__local_str = str_local.file_loader_layout

		self.__layout_colspan = 9

		self.threadpool = threadpool
		self.data_context = data_context

		self.vbox = QGridLayout()
		self.vbox.setContentsMargins(*get_default_margin())
		lbl_instruct = QLabel(self.__local_str['instruction'])
		lbl_note = QLabel(self.__local_str['note'])

		self.progress_placeholder = QVBoxLayout()
		self.preview_placeholder = QVBoxLayout()

		btn_file_dialog = QPushButton(self.__local_str['open_button'])
		btn_file_dialog.pressed.connect(self.open_dialog)

		self.vbox.addWidget(lbl_instruct, 0, 0, self.__layout_colspan, 1, Qt.AlignmentFlag.AlignTop)
		self.vbox.addWidget(lbl_note, 1, 0, self.__layout_colspan, 1, Qt.AlignmentFlag.AlignTop)
		self.vbox.addWidget(btn_file_dialog, 2, 0, self.__layout_colspan, 3, Qt.AlignmentFlag.AlignCenter)

		self.vbox.addLayout(self.progress_placeholder, 5, 0, self.__layout_colspan, 3, Qt.AlignmentFlag.AlignCenter)
		self.vbox.addLayout(self.preview_placeholder, 8, 0, self.__layout_colspan, 9, Qt.AlignmentFlag.AlignCenter)

		self.setLayout(self.vbox)

		self.progressbar = QProgressBar()
		self.progressbar.setValue(0)

	def open_dialog(self):
		fname = QFileDialog.getOpenFileName(self, 'Open file', '${HOME}', 'CSV Files (*.csv)')
		if fname[0]:
			self.load_csv_file(fname[0])

	def load_csv_file(self, filepath: str, sep: str = ',', encoding: str = 'utf-8', chunksize: int = 100000):
		totalrows = sum(1 for row in open(filepath, 'r', encoding=encoding))
		self.progress_placeholder.addWidget(self.progressbar)
		print('Total rows: %d' % totalrows)
		worker = Worker(self.load_data_from_file, filepath, sep, encoding, chunksize, totalrows)
		worker.signals.result.connect(self.on_file_load_complete)
		worker.signals.finished.connect(self.thread_complete)
		worker.signals.progress.connect(self.progress_fn)
		self.threadpool.start(worker)

	def load_data_from_file(
		self, filename: str, sep: str, encoding: str, chunksize: int, totalrows: int, progress_callback
	) -> pd.DataFrame:
		data = []
		prog = 0
		progress_callback.emit(int(prog / totalrows * 100))
		for chunk in pd.read_csv(filename, sep=sep, encoding=encoding, chunksize=chunksize):
			data.append(chunk)
			prog += len(chunk)
			progress_callback.emit(int(prog / totalrows * 100))

		return pd.concat(data)

	def on_file_load_complete(self, datafile: pd.DataFrame):
		self.progressbar.setValue(100)
		self.data_context.add_key_value_pair('dataframe', datafile)
		self.data_context.add_key_value_pair('variables', list(datafile.columns))

	def thread_complete(self):
		self.progress_placeholder.removeWidget(self.progressbar)

	def progress_fn(self, n):
		self.progressbar.setValue(n)
