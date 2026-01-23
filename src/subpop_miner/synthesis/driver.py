import subprocess
from importlib import resources


def run_r_synthesis(input_path, output_path):
	with resources.path('subpop_miner.synthesis', 'worker.R') as r_script_path:
		cmd = ['Rscript', str(r_script_path), input_path, output_path]
		subprocess.run(cmd, check=True)
