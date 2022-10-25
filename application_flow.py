import pandas as pd
from miner.rule import DiscreteVariable, DiscreteVariableDescriptor, RuleMeta, ContinuousVariable, Rule
from miner.frequent_itemsets import FrequentItemsets
import numpy as np

from utils.data_utils import Config


def load_data(filepath, relev_cat):
	all_data = pd.read_csv(filepath, nrows=5000)

	var_list = ['AAGE', 'AANCSTR1', 'AANCSTR2', 'AAUGMENT', 'ABIRTHPL', 'ACITIZEN', 
				'ACLASS', 'ADEPART', 'ADISABL1', 'ADISABL2', 'AENGLISH', 'AFERTIL', 
				'AGE', 'AHISPAN', 'AHOUR89', 'AHOURS', 'AIMMIGR', 'AINCOME1', 
				'AINCOME2', 'AINCOME3', 'AINCOME4', 'AINCOME5', 'AINCOME6', 
				'AINCOME7', 'AINCOME8', 'AINDUSTR', 'ALABOR', 'ALANG1', 'ALANG2', 
				'ALSTWRK', 'AMARITAL', 'AMEANS', 'AMIGSTAT', 'AMOBLLIM', 'AMOBLTY', 
				'ANCSTRY1', 'ANCSTRY2', 'AOCCUP', 'APERCARE', 'APOWST', 'ARACE', 
				'ARELAT1', 'ARIDERS', 'ASCHOOL', 'ASERVPER', 'ASEX', 'ATRAVTME', 
				'AVAIL', 'AVETS1', 'AWKS89', 'AWORK89', 'AYEARSCH', 'AYRSSERV', 
				'CITIZEN', 'CLASS', 'DEPART', 'DISABL1', 'DISABL2', 'ENGLISH', 
				'FEB55', 'FERTIL', 'HISPANIC', 'HOUR89', 'HOURS', 'IMMIGR', 
				'INCOME1', 'INCOME2', 'INCOME3', 'INCOME4', 'INCOME5', 'INCOME6', 
				'INCOME7', 'INCOME8', 'INDUSTRY', 'KOREAN', 'LANG1', 'LANG2', 
				'LOOKING', 'MARITAL', 'MAY75880', 'MEANS', 'MIGPUMA', 'MIGSTATE', 
				'MILITARY', 'MOBILITY', 'MOBILLIM', 'OCCUP', 'OTHRSERV', 'PERSCARE', 
				'POB', 'POVERTY', 'POWPUMA', 'POWSTATE', 'PWGT1', 'RACE', 'RAGECHLD', 
				'REARNING', 'RECTYPE', 'RELAT1', 'RELAT2', 'REMPLPAR', 'RIDERS', 
				'RLABOR', 'ROWNCHLD', 'RPINCOME', 'RPOB', 'RRELCHLD', 'RSPOUSE', 
				'RVETSERV', 'SCHOOL', 'SEPT80', 'SERIALNO', 'SEX', 'SUBFAM1', 
				'SUBFAM2', 'TMPABSNT', 'TRAVTIME', 'VIETNAM', 'WEEK89', 'WORK89', 
				'WORKLWK', 'WWII', 'YEARSCH', 'YEARWRK', 'YRSSERV']

	drop_cols = [var for var in var_list if var not in relev_cat]
	all_data = all_data.drop(columns=drop_cols)

	relat1 = { 
		0: 'Householder',
		1: 'Husband/wife',
		2: 'Son/daughter',
		3: 'Stepson/stepdaughter',
		4: 'Brother/sister',
		5: 'Father/mother',
		6: 'Grandchild',
		7: 'Other relative/Not related',
		8: 'Roomer/boarder/foster child',
		9: 'Housemate/roommate',
		10: 'Unmarried partner',
		11: 'Other nonrelative/Group quarters',
		12: 'Institutionalized person',
		13: 'Other persons in group quarters'
	}
	industry_class = {0: 'Unemployed', 1: 'Agriculture', 2: 'Mining',
						3: 'Construction', 4: 'Manufacturing',
						5: 'Transportation', 6: 'Trade', 7: 'Finance',
						8: 'Service', 9: 'Administration', 10: 'Military',
						11: 'Experience Unemployed'}
	occup_class = {0: 'Unemployed', 1: 'Executive', 2: 'Professional',
					3: 'Technical', 4: 'Service Occupation',
					5: 'Protective Service', 6: 'Farming', 7: 'Precision',
					8: 'Operators', 9: 'Military',
					10: 'Experienced Unemployed'}

	rlabor = {
		0: 'N/A (less than 16 years old)',
		1: 'Civilian employed, at work',
		2: 'Civilian employed, with a job but not at work',
		3: 'Unemployed',
		4: 'Armed forces, at work',
		5: 'Armed forces, with a job but not at work',
		6: 'Not in labor force'
	}

	disbl1 = {
		0: 'N/A (less than 16 years)',
		1: 'Yes, limited in kind or amount of work',
		2: 'No, not limited'
	}

	class_var = {
		0: 'N/A (less than 16 years old/unemployed who never worked/NILF who last worked prior to 1985)',
		1: 'employee of a private for profit company or business or of an individual, for wages, salary, or commissions',
		2: 'Employee of a private not-for-profit, tax-exempt, or charitable organization',
		3: 'Local government employee (city, county, etc.)',
		4: 'State government employee',
		5: 'Federal government employee',
		6: 'Self-employed in own not incorporated business, professional practice, or farm',
		7: 'Self-employed in own incorporated business, professional practice or farm',
		8: 'Working without pay in family business or farm',
		9: 'Unemployed, last worked in 1984 or earlier'
	}

	hour89_var = {
		1: 'Part-time',
		0: 'Full-time'
	}

	week89_var = {
		1: 'Seasonal or Part-Year',
		0: 'Full-time'
	}

	yearsch = {
		0: 'N/a Less Than 3 Yrs. Old',
		1: 'No School Completed',
		2: 'Nursery School',
		3: 'Kindergarten',
		4: '1st, 2nd, 3rd, or 4th Grade',
		5: '5th, 6th, 7th, or 8th Grade',
		6: '9th Grade',
		7: '10th Grade',
		8: '11th Grade',
		9: '12th Grade, No Diploma',
		10: 'High School Graduate, Diploma or Ged',
		11: 'Some Coll., But No Degree',
		12: 'Associate Degree in Coll., Occupational',
		13: 'Associate Degree in Coll., Academic Prog',
		14: 'Bachelors Degree',
		15: 'Masters Degree',
		16: 'Professional Degree',
		17: 'Doctorate Degree'
	}

	cat_dict = {
		'RELAT1': relat1, 'INDUSTRY_CLASS': industry_class,
		'OCCUP_CLASS': occup_class, 'RLABOR': rlabor,
		'DISABL1': disbl1, 'CLASS': class_var, 'HOUR89_CAT': hour89_var,
		'WEEK89_CAT': week89_var, 'YEARSCH': yearsch
	}

	all_data['HOUR89_CAT'] = all_data.apply(lambda row: 1 if row['HOUR89'] < 30 else 0, axis=1)
	all_data['WEEK89_CAT'] = all_data.apply(lambda row: 1 if row['WEEK89'] < 26 else 0, axis=1)
	all_data = all_data[all_data['INCOME1'] > 0]
	all_data = all_data[all_data['AGE'] > 18]
	all_data['INCOME1_CAT'] = all_data.apply(lambda row: 1 if row['INCOME1'] <= 52000 else 0, axis=1)

	return all_data

def get_config(all_data):

	config = {}

	# General Settings
	config['popq1'] = all_data['INCOME1'].quantile(0.25)
	config['popq3'] = all_data['INCOME1'].quantile(0.75)
	config['delta'] =40000
	
	config['pop_iqr'] = config['popq3'] - config['popq1']
	config['pop_threshold'] = config['popq3'] + 3*config['pop_iqr']

	yrsch_min = min(list(all_data['YEARSCH']))
	yrsch_max = max(list(all_data['YEARSCH']))

	# Setting for support and confidence across all algorithms
	# frequent itemset, association rule, genetic algorithm fitness
	config['minsup'] = 0.01
	config['minconf'] = 0.95
	config['max_rulelen'] = 3

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
			# print(list(all_data[col].unique()))
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


def main(filepath):

	relev_cat = ['DISABL1', 'CLASS', 'INDUSTRY_CLASS', 'OCCUP_CLASS', \
		'RLABOR', 'RELAT1']
	relev_cont = ['INCOME1', 'HOUR89', 'AGE', 'WEEK89', 'YEARSCH']
	relev_cat_type = [('DISABL1', 'cat'), ('CLASS', 'cat'), ('INDUSTRY_CLASS', 'cat'), ('OCCUP_CLASS', 'cat'), \
		('RLABOR', 'cat'), ('HOUR89', 'cont'), ('YEARSCH', 'cont'), ('RELAT1', 'cat'), ('WEEK89', 'cont')]
	data = load_data(filepath, relev_cat+relev_cont)
	config = get_config(data)
	rulemeta = build_rulemeta(data, relev_cat_type, 'INCOME1')

	fitemsets = FrequentItemsets(min_support=config['minsup'], max_len=config['max_rulelen'])
	fitemsets.find_frequent_itemsets(data, relev_cat)
	datalen = len(data)
	eval_params = {
				'minsup': config['minsup'],
				'minq1': config['popq1'],
				'minq3': config['popq3'],
				'delta': config['delta']
				}

	# print(fitemsets.itemsets)
	setsizes = sorted(fitemsets.itemsets.keys())
	for setsize in setsizes:
		itemset = fitemsets.itemsets[setsize]
		for fset, freq in itemset.items():
			# print(fset, freq)
			mycats = [fitemsets.id2item[mycat] for mycat in iter(fset)]
			# print(mycats, freq)
			rule = Rule(itemset=mycats, target='INCOME1')
			# rule.build_rule_from_itemset()
			rule_eval = rule.evaluate(data, eval_params, datalen=datalen)
			print('{}: {}'.format(rule.rule_str, 'PASS' if rule_eval else 'FAIL'))
			


if __name__ == "__main__":
	path = '/home/ishahanm/dev/projects/subpop-topcode/census data/USCensus1990Full_industry_occup_coded.csv'
	main(path)