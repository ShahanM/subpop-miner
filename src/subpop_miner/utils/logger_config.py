import logging
import logging.config
import os
import sys
from datetime import datetime

import structlog


def configure_logging(verbose=False, reportpath=None):
	"""
	Configures structlog and standard logging.

	Args:
		verbose (bool): If True, logs will appear in the console (stdout).
		reportpath (str): Custom path for the log file. If None, auto-generates in 'logs/'.
	"""

	# Determine Log File Path
	if reportpath is None:
		if not os.path.exists('logs'):
			os.makedirs('logs')
		# Using .jsonl extension implies structured JSON lines
		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		reportpath = os.path.join('logs', f'log_{timestamp}.jsonl')

	# Define Shared Processors (applied to both file and console)
	shared_processors = [
		structlog.contextvars.merge_contextvars,
		structlog.stdlib.add_logger_name,
		structlog.stdlib.add_log_level,
		structlog.stdlib.PositionalArgumentsFormatter(),
		structlog.processors.TimeStamper(fmt='iso'),
		structlog.processors.StackInfoRenderer(),
		structlog.processors.format_exc_info,  # Handles exceptions nicely
	]

	# Define Handlers (Console vs File)
	handlers = []

	# --- File Handler (Structured JSON) ---
	# We use ProcessorFormatter to render JSON for the file
	file_handler = logging.FileHandler(reportpath)
	file_handler.setFormatter(
		structlog.stdlib.ProcessorFormatter(
			# The foreign_pre_chain processes standard logging calls not made via structlog
			foreign_pre_chain=shared_processors,
			processors=[
				# structlog.processors.remove_keys_from_event_dict(["level", "timestamp"]),
				structlog.processors.JSONRenderer()
			],
		)
	)
	handlers.append(file_handler)

	# --- Console Handler (Pretty Print) ---
	if verbose:
		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setFormatter(
			structlog.stdlib.ProcessorFormatter(
				foreign_pre_chain=shared_processors,
				processors=[
					structlog.dev.ConsoleRenderer()  # Pretty colors and layout
				],
			)
		)
		handlers.append(console_handler)

	# Apply Configuration to Root Logger
	logging.basicConfig(
		level=logging.INFO,
		handlers=handlers,
		force=True,  # Overwrite any existing configs
	)

	# Configure structlog to wrap standard library
	structlog.configure(
		processors=shared_processors
		+ [
			structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
		],
		logger_factory=structlog.stdlib.LoggerFactory(),
		wrapper_class=structlog.stdlib.BoundLogger,
		cache_logger_on_first_use=True,
	)

	return reportpath
