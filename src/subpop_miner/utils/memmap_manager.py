import os
import shutil
import tempfile
import numpy as np
import pandas as pd


class MemmapManager:
	def __init__(self, metadata=None):
		self.temp_dir = None
		# If metadata is provided, we are in 'read' mode (worker)
		if metadata:
			self.metadata = metadata
			self.data = self._load_memmaps()
		else:
			self.metadata = {}
			self.data = {}

	def load_dataframe(self, df: pd.DataFrame):
		"""
		Main Process: Converts a DataFrame to memmap files on disk.
		"""
		self.temp_dir = tempfile.mkdtemp(prefix='miner_memmap_')
		self.metadata = {'temp_dir': self.temp_dir, 'columns': {}, 'len': len(df)}

		for col in df.columns:
			# Determine safe dtype
			series = df[col]
			if pd.api.types.is_numeric_dtype(series):
				dtype = series.dtype
			else:
				# Convert strings/objects to fixed-width unicode
				# Calculate max length in bytes for safety
				max_len = series.astype(str).map(len).max()
				dtype = f'<U{max_len}'

			# Create Memmap file
			filename = os.path.join(self.temp_dir, f'{col}.dat')
			shape = (len(df),)

			# Write mode to populate data
			mm = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
			mm[:] = series.to_numpy().astype(dtype)
			mm.flush()  # Ensure written to disk

			# Store metadata
			self.metadata['columns'][col] = {'dtype': str(dtype), 'shape': shape, 'filename': f'{col}.dat'}

			# Keep a read-only reference for the main process too
			self.data[col] = np.memmap(filename, dtype=dtype, mode='r', shape=shape)

		return self.metadata

	def _load_memmaps(self):
		"""
		Worker Process: Loads memmaps based on metadata.
		"""
		data_map = {}
		base_dir = self.metadata['temp_dir']

		for col, meta in self.metadata['columns'].items():
			path = os.path.join(base_dir, meta['filename'])
			# 'c' mode = Copy-on-write (safe for read-only sharing)
			# 'r' mode = Read-only (safest)
			data_map[col] = np.memmap(path, dtype=meta['dtype'], mode='r', shape=meta['shape'])

		return data_map

	def get_column(self, col_name):
		return self.data[col_name]

	def cleanup(self):
		"""Removes the temporary directory."""
		if self.temp_dir and os.path.exists(self.temp_dir):
			try:
				shutil.rmtree(self.temp_dir)
			except OSError:
				pass
