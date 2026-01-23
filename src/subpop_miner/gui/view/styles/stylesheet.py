DEFAULT_MARGINS = {'left': 3, 'right': 3, 'top': 3, 'bottom': 3}


def get_stylesheet():
	with open('view/styles/styles.css', 'r') as f:
		return f.read()


def get_default_margin():
	return DEFAULT_MARGINS.values()
