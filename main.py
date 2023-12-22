"""
This is the GUI for the sub-population outlier detection algorithm. For more 
information, please refer to the following paper:
<Paper currently under review.>

For the command line tool refer to the readme.md file.

@Author: Mehtab "Shahan" Iqbal
@Affiliation: School of Computing, Clemson University

@Disclaimer: This code is provided as is, without any guarantees of any kind.
	The author is not responsible for any damages resulting from the use of this
	code.

@License: This code is free to use under the MIT license 
	(https://opensource.org/licenses/MIT) as long as the authorship is properly
	acknowledged.
"""
import sys

from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from view.styles.stylesheet import styles

from view.main_window import MainWindow


if __name__ == '__main__':
	app = QApplication(sys.argv)
	app.setStyleSheet(styles)
	w = MainWindow()
	app.exec()
