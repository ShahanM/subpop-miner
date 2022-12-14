import pickle as pkl
import time

from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import pandas as pd

from miner.frequent_itemsets import FrequentItemsets
from miner.optimizer import GeneticOptimizer
from miner.rule import (ContinuousVariable, DiscreteVariableDescriptor, Rule,
                        RuleMeta, DiscreteVariable)
from miner.utils import *
from utils.data_utils import Config


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

	all_data['HOUR89_CAT'] = all_data.apply(lambda row: 1 if row['HOUR89'] < 30 else 0, axis=1)
	all_data['WEEK89_CAT'] = all_data.apply(lambda row: 1 if row['WEEK89'] < 26 else 0, axis=1)
	all_data = all_data[all_data['INCOME1'] > 0]
	all_data = all_data[all_data['AGE'] > 18]
	all_data['INCOME1_CAT'] = all_data.apply(lambda row: 1 if row['INCOME1'] <= 52000 else 0, axis=1)

	all_data['HOUR_INV'] = all_data.apply(lambda row: 99 - row['HOUR89'], axis=1)

	return all_data

def get_config(all_data):

	config = {}

	# General Settings
	config['delta'] =40000
	config['delta2'] = 20000

	# Setting for support and confidence across all algorithms
	# frequent itemset, association rule, genetic algorithm fitness
	config['minsup'] = 0.01
	config['max_rulelen'] = 3
	# config['minconf'] = 0.95

	# Settings for Genetic Algorithm
	config['num_population'] = 250
	config['num_generations'] = 100
	config['crossover_rate'] = 0.5
	config['mutation_rate'] = 0.4

	return config


def build_rulemeta(all_data, relevant_col_types, target_col):
	
	discvars = {}
	contvars = {}
	for col, coltype in relevant_col_types:
		if coltype == 'cat':
			var = DiscreteVariableDescriptor(name=col, values=sorted(list(all_data[col].unique())))
			discvars[col] = var
		elif coltype == 'cont':
			lbound = all_data[col].min()
			ubound = all_data[col].max()
			corref = np.corrcoef(all_data[col], all_data[target_col])[0,1]
			var = ContinuousVariable(name=col, lbound=lbound, ubound=ubound, correlation=corref)
			contvars[col] = var
	
	rulemeta = RuleMeta(continuous_vars=contvars, discrete_vars=discvars)
	
	return rulemeta

def run_optimizer(optimizer, rule, eval_params, datalen, i, total):
	print(f'Running optimizer for rule {i} of {total}')
	start = time.time()
	newrule = optimizer.optimize(rule, eval_params)
	# return newrule
	if newrule.evaluate(optimizer.data, eval_params, datalen=datalen):
		print(f'Rule {i} of {total} improved in {time.time() - start} seconds')
		return newrule
	else:
		print(f'Rule {i} of {total} did not improve in {time.time() - start} seconds')
		return None


def parallel_gen(optimizer, rules, num_processes, eval_params, datalen):
	with Pool(processes=num_processes) as pool:
		results = pool.starmap(run_optimizer, [(optimizer, rule, eval_params, datalen, i, len(rules)) for i, rule in enumerate(rules, 1)])

		return [r for r in results if r is not None]


def main(filepath):
	starttime = time.time()
	print('Starting demo for rule mining on the 1990 Census Income Dataset')
	print('Preparing to load data...')
	
	startsubtime = time.time()
	totalrows = sum(1 for _ in open(filepath, 'r'))
	print('Time taken to count rows: {} seconds'.format(time.time() - startsubtime))
	print('------------------\n')
	
	print('Loading total rows: %d' % totalrows)
	startsubtime = time.time()
	relev_cat = ['DISABL1', 'CLASS', 'INDUSTRY_CLASS', 'OCCUP_CLASS', \
		'RLABOR', 'RELAT1']

	relev_cont = ['INCOME1', 'HOUR89', 'AGE', 'WEEK89', 'YEARSCH']

	relev_cat_type = [('DISABL1', 'cat'), ('CLASS', 'cat'), ('INDUSTRY_CLASS', 'cat'), ('OCCUP_CLASS', 'cat'), \
		('RLABOR', 'cat'), ('HOUR89', 'cont'), ('YEARSCH', 'cont'), ('RELAT1', 'cat'), ('WEEK89', 'cont'), ('HOUR_INV', 'cont')]
	# relev_cat_type = [('DISABL1', 'cat'), ('CLASS', 'cat'), ('INDUSTRY_CLASS', 'cat'), ('OCCUP_CLASS', 'cat'), \
	# 	('RLABOR', 'cat'), ('HOUR89_CAT', 'cat'), ('YEARSCH', 'cont'), ('RELAT1', 'cat'), ('WEEK89_CAT', 'cat')]
	
	data = load_data(filepath, relev_cat+relev_cont, chunksize=100000, totalrows=totalrows)
	# data = load_data(filepath, relev_cat+relev_cont, rowstoload=5)
	print('Time taken to load data: {} seconds'.format(time.time() - startsubtime))
	print('------------------\n')

	print('Preparing data for rule mining...')
	startsubtime = time.time()
	print('Loading config...')
	config = get_config(data)
	print('Generating rule metadata...')
	print('Loaded config with the following settings:')
	print('General Settings ------------------')
	print('Target threshold delta: %d' % config['delta'])
	print('Pruning parameter delta2: %d' % config['delta2'])
	print('Minimum support: %f' % config['minsup'])
	print('Maximum rule length: %d' % config['max_rulelen'])
	print('Genetic Algorithm Settings --------')
	print('Population size: %d' % config['num_population'])
	print('Number of generations: %d' % config['num_generations'])
	print('Crossover rate: %f' % config['crossover_rate'])
	print('Mutation rate: %f' % config['mutation_rate'])
	rulemeta = build_rulemeta(data, relev_cat_type, 'INCOME1')
	print('Relevant variable summary ------------------')
	print('Discrete variables:')
	for i, (varname, var) in enumerate(rulemeta.discrete_vars.items()):
		print('{}. {}: categories={}'.format(i, var.name, var.values))
	print('Continuous variables:')
	for i, (varname, var) in enumerate(rulemeta.continuous_vars.items()):
		print('{}. {}: lbound={}, ubound={}, correlation={}'.format(i, var.name, var.lbound, var.ubound, var.correlation))
	print('Time taken to prepare data: {} seconds'.format(time.time() - startsubtime))
	print('------------------\n')
	
	print('Starting rule mining...')
	startsubtime = time.time()
	print('Starting frequent itemset mining...')
	fitemsets = FrequentItemsets(min_support=config['minsup'], max_len=config['max_rulelen'])
	fitemsets.find_frequent_itemsets(data, relev_cat)
	num_fitemsets = sum([len(itemst) for itemst in fitemsets.itemsets.values()])
	print('Found %d frequent itemsets' % num_fitemsets)
	print('Time taken to mine frequent itemsets: {} seconds'.format(time.time() - startsubtime))
	print('------------------\n')

	print('Evaluating itemsets for outlier...')
	startsubtime = time.time()
	datalen = len(data)
	config['popq1'] = data['INCOME1'].quantile(0.25)
	config['popq3'] = data['INCOME1'].quantile(0.75)
	eval_params = {
				'minsup': config['minsup'],
				'minq1': config['popq1'],
				'minq3': config['popq3'],
				'delta': config['delta']
				}

	passlist = []
	faillist = []

	seen = []
	setsizes = sorted(fitemsets.itemsets.keys())
	for setsize in tqdm(setsizes): # TODO parallelize this
		itemset = fitemsets.itemsets[setsize]
		for fset, freq in itemset.items():
			mycats = [fitemsets.id2item[mycat] for mycat in iter(fset)]
			# print(mycats)
			dvars = {dvar[0]: DiscreteVariable(name=dvar[0], value=dvar[1]) for dvar in map(lambda x: x.split('='), mycats)}
			rule = Rule(itemset=mycats, target='INCOME1', discrete_vars=dvars)
			print(rule)
			rule_eval = rule.evaluate(data, eval_params, datalen=datalen)

			if rule_eval:
				rset = set(rule.itemset)
				if not skiprule(rset, seen):
					seen.append(rset)
					passlist.append(rule)
			else:
				if rule.support >= config['minsup']:
					faillist.append(rule)
				else: continue

	print('{} subpopulation rules skipped'.format(len(seen)))
	print('{} subpupulation that PASSED outlier evaluation.'.format(len(passlist)))
	print('{} subpupulation that FAILED outlier evaluation.'.format(len(faillist)))
	print('Time taken to evaluate itemsets: {} seconds'.format(time.time() - startsubtime))
	print('------------------\n')

	print('Pruning passlist and removing overlapping subpulation rules...')
	startsubtime = time.time()
	finallist = delta_prune_empty(passlist, config['delta2'])
	print('{} subpopulation rules after pruning'.format(len(finallist)))
	print('Time taken to prune passlist: {} seconds'.format(time.time() - startsubtime))
	print('------------------\n')

	print('Pruning list of failed subpopulation rules before adding continuous variables...')
	startsubtime = time.time()
	gencandidates = delta_prune_empty(faillist, config['delta2'])
	gencandidates = delta_prune(finallist, gencandidates, 20000)
	print('{} subpopulation rules after pruning'.format(len(gencandidates)))
	print('Time taken to prune faillist: {} seconds'.format(time.time() - startsubtime))
	print('------------------\n')

	print('Adding continuous variables to rules with Genetic Algorithm...')
	startsubtime = time.time()

	# INFLATED WITH TEMPLATES
	testcand = gencandidates.copy()
	gencandidates = []
	for rule in testcand:
		for cvar in rulemeta.continuous_vars.values():
			newrule = Rule(itemset=rule.itemset[:], target='INCOME1', \
				discrete_vars=rule.discrete_vars.copy())
			ncvar = ContinuousVariable(name=cvar.name, lbound=cvar.lbound, \
				ubound=cvar.ubound, correlation=cvar.correlation)
			newrule.add_continuous_variable(ncvar)
			gencandidates.append(newrule)
		
		cvars = list(rulemeta.continuous_vars.keys())
		for cvaridx1 in range(len(cvars)-1):
			for cvaridx2 in range(cvaridx1+1, len(cvars)):
				newrule = Rule(itemset=rule.itemset[:], target='INCOME1', \
					discrete_vars=rule.discrete_vars.copy())
				ncvar1 = ContinuousVariable(name=rulemeta.continuous_vars[cvars[cvaridx1]].name, \
					lbound=rulemeta.continuous_vars[cvars[cvaridx1]].lbound, \
					ubound=rulemeta.continuous_vars[cvars[cvaridx1]].ubound, \
					correlation=rulemeta.continuous_vars[cvars[cvaridx1]].correlation)
				ncvar2 = ContinuousVariable(name=rulemeta.continuous_vars[cvars[cvaridx2]].name, \
					lbound=rulemeta.continuous_vars[cvars[cvaridx2]].lbound, \
					ubound=rulemeta.continuous_vars[cvars[cvaridx2]].ubound, \
					correlation=rulemeta.continuous_vars[cvars[cvaridx2]].correlation)
				newrule.add_continuous_variable(ncvar1)
				newrule.add_continuous_variable(ncvar2)
				gencandidates.append(newrule)
		newrule = Rule(itemset=rule.itemset[:], target='INCOME1', \
			discrete_vars=rule.discrete_vars.copy())
		for cvar in rulemeta.continuous_vars.values():
			ncvar = ContinuousVariable(name=cvar.name, lbound=cvar.lbound, \
				ubound=cvar.ubound, correlation=cvar.correlation)
			newrule.add_continuous_variable(ncvar)
		gencandidates.append(newrule)

	# COMBINATION TEMPLATES
	# gencandidates = []
	for cvar in rulemeta.continuous_vars.values():
		newrule = Rule(itemset=[], target='INCOME1')
		ncvar = ContinuousVariable(name=cvar.name, lbound=cvar.lbound, \
			ubound=cvar.ubound, correlation=cvar.correlation)
		newrule.add_continuous_variable(ncvar)
		gencandidates.append(newrule)
	
	cvars = list(rulemeta.continuous_vars.keys())
	for cvaridx1 in range(len(cvars)-1):
		for cvaridx2 in range(cvaridx1+1, len(cvars)):
			newrule = Rule(itemset=[], target='INCOME1')
			ncvar1 = ContinuousVariable(name=rulemeta.continuous_vars[cvars[cvaridx1]].name, \
				lbound=rulemeta.continuous_vars[cvars[cvaridx1]].lbound, \
				ubound=rulemeta.continuous_vars[cvars[cvaridx1]].ubound, \
				correlation=rulemeta.continuous_vars[cvars[cvaridx1]].correlation)
			ncvar2 = ContinuousVariable(name=rulemeta.continuous_vars[cvars[cvaridx2]].name, \
				lbound=rulemeta.continuous_vars[cvars[cvaridx2]].lbound, \
				ubound=rulemeta.continuous_vars[cvars[cvaridx2]].ubound, \
				correlation=rulemeta.continuous_vars[cvars[cvaridx2]].correlation)
			newrule.add_continuous_variable(ncvar1)
			newrule.add_continuous_variable(ncvar2)
			gencandidates.append(newrule)
			# newrule.add_continuous_variable(rulemeta.continuous_vars[cvars[cvaridx1]].copy())
			# newrule.add_continuous_variable(rulemeta.continuous_vars[cvars[cvaridx2]].copy())
			# gencandidates.append(newrule)
			# tt.append(newrule)
	newrule = Rule(itemset=[], target='INCOME1')
	for cvar in rulemeta.continuous_vars.values():
		ncvar = ContinuousVariable(name=cvar.name, lbound=cvar.lbound, \
			ubound=cvar.ubound, correlation=cvar.correlation)
		newrule.add_continuous_variable(ncvar)
		# newrule.add_continuous_variable(cvar.copy())
	gencandidates.append(newrule)

	# INDIVIDUAL TEMPLATES
	# emptyrule = Rule(itemset=[], target='INCOME1')
	# gencandidates.append(emptyrule)
	# for cvar in rulemeta.continuous_vars.values():
	# 	newrule = Rule(itemset=[], target='INCOME1')
	# 	fcvar = ContinuousVariable(name=cvar.name, lbound=cvar.lbound, \
	# 		ubound=cvar.ubound, correlation=cvar.correlation, freeze=True)
	# 	newrule.add_continuous_variable(fcvar)
	# 	gencandidates.append(newrule)

	genalgo = GeneticOptimizer(population_size=config['num_population'],\
		generations=config['num_generations'],
		crossover_rate=config['crossover_rate'], 
		mutation_rate=config['mutation_rate'], 
		rulemeta=rulemeta, data=data, 
		aggressive_mutation=True, parallel=False, \
		force_int_bounds=True)

	# quant = []
	# for rule in gencandidates: # TODO - parallelize this
	# 	newrule = genalgo.optimize(rule, eval_params)
	# 	if newrule.evaluate(data, eval_params, datalen=datalen):
	# 		quant.append(newrule)
	print("Running genetic algorithm on {} rules".format(len(gencandidates)))
	quant = parallel_gen(genalgo, gencandidates, 8, eval_params, datalen)

	print('Found {} outlier subpopulation with continuous variables.'.format(len(quant)))
	print('Time taken to add continuous variables: {} seconds'.format(time.time() - startsubtime))
	print('------------------\n')

	print('Pruning list of subpopulation rules with continuous variables...')
	startsubtime = time.time()
	# cleanquant = delta_prune_empty(quant, config['delta2'])
	# finalquant = delta_prune(finallist, cleanquant, 20000)
	print('Removing rules with continuous variables that span the entire range...')
	finalquant = prune_spanning_vars(quant, rulemeta)
	print('Found {} outlier subpopulation with continuous variables after removing spanning variables.'.format(len(finalquant)))
	print('Removing duplicate rules...')
	# finalquant = remove_duplicate_rules(finalquant)
	# print('Found {} outlier subpopulation with continuous variables after removing duplicates.'.format(len(finalquant)))
	print('Time taken to prune quant: {} seconds'.format(time.time() - startsubtime))
	print('------------------\n')

	print('Merging all subpopulation rules and writing to file...')
	startsubtime = time.time()
	merged = finallist + finalquant
	# merged = finallist + quant
	# merged = quant

	with open('mergedlst.pkl', 'wb') as f:
		pkl.dump(merged, f)
	
	print('Total subpopulation rules found: {}'.format(len(merged)))
	print('Time taken to merge rules: {} seconds'.format(time.time() - startsubtime))
	print('Total time taken: {} seconds'.format(time.time() - starttime))
	print('------------------\n')


if __name__ == "__main__":
	path = '/home/ishahanm/dev/projects/other_research/subpop-topcode/census data/USCensus1990Full_industry_occup_coded.csv'
	# main(path)
	# pruned = []
	# with open('mergedlst.pkl', 'rb') as f:
		# merged = pkl.load(f)

		# pruned = remove_duplicate_rules(merged[10:])

	# with open('pruned.pkl', 'wb') as f:
		# pkl.dump(pruned, f)
	
	relev_cat = ['DISABL1', 'CLASS', 'INDUSTRY_CLASS', 'OCCUP_CLASS', \
		'RLABOR', 'RELAT1']

	relev_cont = ['INCOME1', 'HOUR89', 'AGE', 'WEEK89', 'YEARSCH']
	totalrows = sum(1 for _ in open(path, 'r'))
	mydf = load_data(path, relev_cat+relev_cont, chunksize=10000, totalrows=totalrows)

	print(mydf.shape)
