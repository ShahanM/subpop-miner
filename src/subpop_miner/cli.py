import multiprocessing
import os
import pickle as pkl
import time
from datetime import datetime

import numpy as np
import pandas as pd
import structlog
from subpop_miner.miner.frequent_itemsets import FrequentItemsets
from subpop_miner.miner.optimizer import GeneticOptimizer
from subpop_miner.miner.rule import ContinuousVariable, DiscreteVariable, DiscreteVariableDescriptor, Rule, RuleMeta
from subpop_miner.miner.utils import (
	delta_prune,
	delta_prune_empty,
	inflate_with_quantitative_templates,
	prune_spanning_vars,
	remove_duplicate_rules,
)
from subpop_miner.utils.memmap_manager import MemmapManager
from tqdm import tqdm
from subpop_miner.utils.logger_config import configure_logging

from subpop_miner.utils.print_utils import export_rules, generate_report

# Initialize configuration once at the start
if not os.path.exists('logs'):
	os.makedirs('logs')
reportpath = os.path.join('logs', 'log_{}.txt'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))
verbose = True
actual_log_path = configure_logging(verbose=verbose, reportpath=reportpath)

# Get the logger
logger = structlog.get_logger()

DATA_CACHE = None
FREQUENT_ITEMS_CACHE = None

# 1. Define a global variable to hold the optimizer in each worker process
worker_optimizer = None


# 2. Define an initializer function that runs once when each process starts
def init_worker(optimizer):
	global worker_optimizer
	worker_optimizer = optimizer
	worker_optimizer.hydrate()
	worker_optimizer.shared_df = pd.DataFrame(worker_optimizer.data_manager.data, copy=False)


def load_data(filepath, relev_cat, chunksize=10000, totalrows=None, rowstoload=None):
	if totalrows:
		data = []
		if rowstoload:
			totalrows = min(rowstoload, totalrows)
			chunksize = min(chunksize, rowstoload)

		with tqdm(total=totalrows) as pbar:
			for chunk in tqdm(pd.read_csv(filepath, chunksize=chunksize, low_memory=False)):
				data.append(chunk)
				pbar.update(chunksize)

		all_data = pd.concat(data)
	else:
		all_data = pd.read_csv(filepath, nrows=rowstoload)

	all_data = all_data[relev_cat]

	return all_data


def get_config():
	_config = {}

	# General Settings
	_config['delta1'] = 30000
	_config['delta2'] = 10000
	# config['delta2'] = 10000

	# Setting for support and confidence across all algorithms
	# frequent itemset, association rule, genetic algorithm fitness
	_config['minsup'] = 0.005
	_config['max_rulelen'] = 3
	# config['minconf'] = 0.95

	# Settings for Genetic Algorithm
	_config['num_population'] = 250
	_config['num_generations'] = 100
	_config['crossover_rate'] = 0.5
	_config['mutation_rate'] = 0.4

	return _config


def build_rulemeta(all_data, relevant_col_types, target_col):
	discvars = {}
	contvars = {}
	for col, coltype in relevant_col_types:
		if coltype == 'cat':
			var = DiscreteVariableDescriptor(name=col, values=sorted(all_data[col].unique()))
			discvars[col] = var
		elif coltype == 'cont':
			lbound = all_data[col].min()
			ubound = all_data[col].max()
			corref = np.corrcoef(all_data[col], all_data[target_col])[0, 1]
			var = ContinuousVariable(name=col, lbound=lbound, ubound=ubound, correlation=corref)
			contvars[col] = var

	rulemeta = RuleMeta(continuous_vars=contvars, discrete_vars=discvars)

	return rulemeta


def run_optimizer_worker(rule, eval_params, datalen, i, total):
	start = time.time()

	# Use the global optimizer instance specific to this process
	newrule = worker_optimizer.optimize(rule, eval_params)

	if newrule.evaluate(worker_optimizer.shared_df, eval_params, datalen=datalen):
		return newrule, time.time() - start
	else:
		return None, time.time() - start


def parallel_gen(optimizer, rules, num_processes, eval_params, datalen):
	passedresults = []
	durations = []

	# Pass the optimizer logic ONCE per process via initializer
	with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(optimizer,)) as pool:
		# Prepare arguments (Notice: optimizer is NOT in the list anymore)
		task_args = [(rule, eval_params, datalen, i, len(rules)) for i, rule in enumerate(rules, 1)]

		results = pool.starmap(run_optimizer_worker, task_args)

		passedresults = [r for (r, _) in results if r is not None]
		durations = [t for (_, t) in results]

	return passedresults, durations


def extract_rules(filepath, output, config, verbose=False, report=False, reportpath=None):
	global DATA_CACHE
	global FREQUENT_ITEMS_CACHE
	starttime = time.time()

	logger.info('Starting demo for rule mining', dataset='USCensus1990')
	logger.info('Preparing to load data...')
	startsubtime = time.time()

	relev_cat = ['DISABL1', 'CLASS', 'INDUSTRY_CLASS', 'OCCUP_CLASS', 'RLABOR', 'RELAT1']

	relev_cont = ['INCOME1', 'HOUR89', 'AGE', 'WEEK89', 'YEARSCH']
	target_col = 'INCOME1'

	relev_cat_type = [
		('DISABL1', 'cat'),
		('CLASS', 'cat'),
		('INDUSTRY_CLASS', 'cat'),
		('OCCUP_CLASS', 'cat'),
		('RLABOR', 'cat'),
		('HOUR89_CAT', 'cat'),
		('YEARSCH', 'cont'),
		('RELAT1', 'cat'),
		('WEEK89_CAT', 'cat'),
	]

	if DATA_CACHE is None:
		totalrows = sum(1 for _ in open(filepath))
		logger.info('Time taken to count rows', duration_seconds=time.time() - startsubtime)
		logger.info('Loading data', total_rows=totalrows)
		startsubtime = time.time()

		data = load_data(filepath, relev_cat + relev_cont, chunksize=100000, totalrows=totalrows)

		logger.info('Time taken to load data', duration_seconds=time.time() - startsubtime)
		DATA_CACHE = data
	else:
		logger.info('Data already loaded, using cached data')
		data = DATA_CACHE

	logger.info('Preparing data for rule mining')
	startsubtime = time.time()
	data['HOUR89_CAT'] = np.where(data['HOUR89'] < 30, 1, 0)
	data['WEEK89_CAT'] = np.where(data['WEEK89'] < 26, 1, 0)

	relev_cat.append('HOUR89_CAT')
	relev_cat.append('WEEK89_CAT')

	data = data[data['INCOME1'] > 0]
	data = data[data['AGE'] > 18]
	logger.info('Time taken to filter data', duration_seconds=time.time() - startsubtime)

	startsubtime = time.time()
	logger.info('Generating rule metadata.')
	logger.info(
		'Configuration loaded',
		target_threshold_delta1=config['delta1'],
		pruning_parameter_delta2=config['delta2'],
		minimum_support=config['minsup'],
		population=config['num_population'],
		generation=config['num_generations'],
		crossover_rate=config['crossover_rate'],
		mutation_rate=config['mutation_rate'],
	)

	rulemeta = build_rulemeta(data, relev_cat_type, target_col)

	# Create lists of dicts representation
	discrete_summary = [{'name': v.name, 'values': v.values} for v in rulemeta.discrete_vars.values()]

	continuous_summary = [
		{'name': v.name, 'lbound': v.lbound, 'ubound': v.ubound, 'corr': v.correlation}
		for v in rulemeta.continuous_vars.values()
	]

	logger.info('Relevant variable summary', discrete_vars=discrete_summary, continuous_vars=continuous_summary)
	logger.info('Time taken to prepare data', duration_seconds=time.time() - startsubtime)
	startsubtime = time.time()
	logger.info('Starting frequent itemset mining')

	fitemsets = None
	if FREQUENT_ITEMS_CACHE is None:
		fitemsets = FrequentItemsets(min_support=config['minsup'], max_len=config['max_rulelen'])
		fitemsets.find_frequent_itemsets(data, relev_cat)
		FREQUENT_ITEMS_CACHE = fitemsets
		num_fitemsets = sum([len(itemst) for itemst in fitemsets.itemsets.values()])
		logger.info('Frequent itemsets', num_itemset=num_fitemsets)
		logger.info('Time taken to mine frequent itemsets', duration_seconds=time.time() - startsubtime)
	else:
		logger.info('Found frequent itemsets in cache')
		fitemsets = FREQUENT_ITEMS_CACHE

	logger.info('Evaluating itemsets for outliers')
	startsubtime = time.time()

	datalen = len(data)
	config['popq1'] = data[target_col].quantile(0.25)
	config['popq3'] = data[target_col].quantile(0.75)
	eval_params = {
		'minsup': config['minsup'],
		'minq1': config['popq1'],
		'minq3': config['popq3'],
		'delta1': config['delta1'],
	}

	passlist = []
	faillist = []
	seen = []
	setsizes = sorted(fitemsets.itemsets.keys())

	def catval_splitter(catlst):
		for catval in catlst:
			yield catval.split('=')

	qualtimedata = []
	for setsize in setsizes:
		itemset = fitemsets.itemsets[setsize]
		logger.info('Set size', set_size=setsize, total_itemset=len(itemset.items()))
		for fset, _ in tqdm(itemset.items()):
			ruletime = time.time()
			mycats = [fitemsets.id2item[mycat] for mycat in iter(fset)]
			dvars = {dvar[0]: DiscreteVariable(name=dvar[0], value=dvar[1]) for dvar in catval_splitter(mycats)}
			rule = Rule(itemset=mycats, target='INCOME1', discrete_vars=dvars)
			rule_eval = rule.evaluate(data, eval_params, datalen=datalen)

			qualtimedata.append(time.time() - ruletime)
			if rule_eval:
				passlist.append(rule)
			else:
				if rule.support >= config['minsup']:
					faillist.append(rule)
				else:
					continue

	with open(f'{config["delta1"]}_{config["delta2"]}_qual_time_data.txt', 'w') as f:
		f.write('\n'.join(map(str, qualtimedata)))

	logger.info('Evaluation summary')
	logger.info('Skipped', skipped=len(seen))
	logger.info('Passed', passed=len(passlist))
	logger.info('Failed', failed=len(faillist))
	logger.info('Time taken to evaluate itemsets', duration_seconds=time.time() - startsubtime)
	logger.info('Pruning passlist and removing overlapping rules')

	startsubtime = time.time()

	# Prune the passlist to remove nested rules
	finallist = delta_prune_empty(passlist, config['delta2'])
	logger.info('Rules after pruning', pruned_passed=len(finallist))
	logger.info('Time taken to prune passlist', duration_seconds=time.time() - startsubtime)
	logger.info('Pruning list of failed subpopulation rules before adding continuous variables')
	startsubtime = time.time()

	# Prune the failed list to remove nested rules
	gencandidates = delta_prune_empty(faillist, config['delta2'])

	logger.info('Candidates after pruning', gen_candidates=(len(gencandidates)))

	# Prune the failed list to remove nested rules shared with the passlist
	gencandidates = delta_prune(finallist, gencandidates, config['delta2'])

	logger.info('Candidates after prunning against pass list', gen_candidates=(len(gencandidates)))
	logger.info('Time taken to prune faillist', duration_seconds=time.time() - startsubtime)
	logger.info('Adding continuous variables to candidates')
	startsubtime = time.time()

	gencandidates = inflate_with_quantitative_templates(gencandidates, rulemeta, target_col)

	logger.info('Converting data to shared memory maps for parallel optimization.')
	mem_mgr = MemmapManager()
	memmap_metadata = mem_mgr.load_dataframe(data)

	finalquant = []
	try:
		genalgo = GeneticOptimizer(
			population_size=config['num_population'],
			generations=config['num_generations'],
			crossover_rate=config['crossover_rate'],
			mutation_rate=config['mutation_rate'],
			rulemeta=rulemeta,
			data_metadata=memmap_metadata,
			aggressive_mutation=True,
			parallel=False,
			force_int_bounds=True,
		)

		logger.info('Running genetic algorithm')
		quant, qdur = parallel_gen(genalgo, gencandidates, 5, eval_params, datalen)

		with open(f'{config["delta1"]}_{config["delta2"]}_quant_duration.txt', 'w') as f:
			f.write('\n'.join(map(str, qdur)))

		logger.info('Rules with continuous variables', num_cont_rules=len(quant))
		logger.info('Time taken to add continuous variables', duration_seconds=time.time() - startsubtime)

		logger.info('Removing duplicates, spanning range, and pruning')
		startsubtime = time.time()
		finalquant = prune_spanning_vars(quant, rulemeta)

		logger.info('Remaining after removing spanning range', num_cont_rules=len(finalquant))

		finalquant = remove_duplicate_rules(finalquant)

		logger.info('Remaining after removing duplicates', num_cont_rules=len(finalquant))

		finalquant = delta_prune_empty(finalquant, config['delta2'])

		logger.info('Remaining after pruning', num_cont_rules=len(finalquant))
		logger.info('Time taken to prune', duration_seconds=time.time() - startsubtime)
	finally:
		logger.info('Cleaning up shared memory.')
		mem_mgr.cleanup()

	logger.info('Merging all subpopulation rules and writing to file.')

	# Prune the final quant list to remove nested rules
	merged = finallist + finalquant

	with open(output, 'wb') as f:
		pkl.dump(merged, f)

	logger.info('Total rules found', total_rules=len(merged))
	logger.info(
		'Total time taken',
		delta1=config['delta1'],
		delta2=config['delta2'],
		duration_seconds=time.time() - starttime,
	)


def run_delta_experiments(data_path, config, filedatestr: str, base_out_dir: str):
	experimental_values = {'delta1': [30000, 40000, 50000], 'delta2': [0, 10000, 15000, 20000, float('inf')]}
	logger.info(
		'Start experiment suite',
		delta1_values=experimental_values['delta1'],
		delta2_values=experimental_values['delta2'],
	)
	for delta1 in experimental_values['delta1']:
		for delta2 in experimental_values['delta2']:
			config['delta1'] = delta1
			config['delta2'] = delta2
			logger.info('Running experiment', delta1=delta1, delta2=delta2)

			div = 1000
			if delta2 == float('inf'):
				file_mid = f'_{filedatestr}_sup005d1{int(delta1 / div)}kd2inf'
			else:
				file_mid = f'_{filedatestr}_sup005d1{int(delta1 / div)}kd2{int(delta2 / div)}k'

			output_filename = f'{base_out_dir}mergedlst{file_mid}.pkl'
			extract_rules(data_path, f'{output_filename}', config, verbose=True, report=True)

			with open(output_filename, 'rb') as f:
				mergedlist = pkl.load(f)
				lendict = {}
				for rule in mergedlist:
					updated = rule.copy()
					updated.itemset = sorted(updated.itemset)
					lendict.setdefault(len(updated.itemset), []).append(updated)

				finallist = []
				for key in sorted(lendict.keys()):
					finallist.extend(sorted(lendict[key], key=lambda x: x.rule_str, reverse=True))

				# sortedlist = sorted(mergedlist, key=lambda x: x.rule_str, reverse=True)
				generate_report(f'{base_out_dir}report{file_mid}.txt', finallist)
				export_rules(f'{base_out_dir}rules{file_mid}.txt', finallist)


def main():
	config = get_config()
	run_delta_experiments(path, config, '2026_01_22', 'runtime/results/census1990/2026/')


if __name__ == '__main__':
	config = get_config()

	run_delta_experiments(path, config, '2026_01_22', 'runtime/results/census1990/2026/')
