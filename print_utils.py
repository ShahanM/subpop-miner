import pickle

from miner.rule import Rule, RuleMeta
from typing import List


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


# def write_rules_file(ruleslist, fname, popthresh):
#     with open(fname, 'w') as fptr:
#         fptr.write('Total Rules: {}'.format(len(ruleslist)))
#         fptr.write('\n\n')
#         for rule_ in ruleslist:
#             sblist = []
#             skip = False
#             for k, v in rule_['rule_dict'].items():
#                 if k in cat_dict.keys():
#                     if type(v) == dict:
#                         if int(v['ubound']) <= 3:
#                             print(rule_)
#                             skip = True
#                         if int(v['ubound']) == 4 and len(rule_['rule_dict'].keys()) > 1:
#                             skip = True
#                         sblist.append('{}: {}'.format(k, cat_dict[k][int(v['ubound'])]))
#                     else:
#                         sblist.append('{}: {}'.format(k, cat_dict[k][v]))
#             subpop = ' AND '.join(sblist)
#             # subpop = ' AND '.join(['{}: {}'.format(k, cat_dict[k][v]) \
#             #     for k, v in rule_['rule_dict'].items() if k in cat_dict.keys()])
#             q1 = rule_['q1']
#             q3 = rule_['q3']
#             mu=3
#             threshold = rule_['threshold']

#             conf = rule_['confidence']
#             supp = rule_['support']
#             rulestr = rule_['rule_str']
            
#             filter_status = ''
#             if rule_['threshold'] <= popthresh:
#                 filter_status = 'PASS'
#             else:
#                 filter_status = 'FAIL'
#                 continue

#             if skip:
#                 continue

#             pretty_print_to_file(rule_string=rulestr, confidence=conf, \
#                 support=supp, subpop=subpop, filter_status=filter_status, \
#                 q1=q1, q3=q3, mu=mu, threshold=threshold, fileptr=fptr)


# def pretty_print_to_file(rule_string, confidence, support, subpop, \
#     filter_status, q1, q3, mu, threshold, fileptr):	
#     fileptr.write('\nShowing for subpopulation {}\n'.format(rule_string))
#     fileptr.write('Confidence {}\n'.format(confidence))
#     fileptr.write('Support {}\n'.format(support))
#     fileptr.write('{}\n'.format(subpop))
#     fileptr.write('-----------------------------------------------------------\n')
#     fileptr.write('Threshold test: ' + filter_status + '\n')
#     fileptr.write('Quartile 1: {}\n'.format(q1))
#     fileptr.write('Quartile 3: {}\n'.format(q3))
#     fileptr.write('IQR: {}\n'.format(q3 - q1))
#     fileptr.write('Q3 + {} *IQR: {}\n'.format(mu, threshold))
#     fileptr.write('Population Threshold: {}\n'.format(pop_threshold))
#     fileptr.write('===========================================================\n\n')




def generate_report(filename, rules: List[Rule]):
	with open(filename, 'w') as fptr:
		fptr.write('Total Rules: {}'.format(len(rules)))
		fptr.write('\n\n')
		for rule in rules:
			fptr.write('\nShowing for subpopulation {}\n'.format(rule.rule_str))
			fptr.write('Support {}\n'.format(rule.support))
			sblist = []
			for item in rule.itemset:
				if '=' in item:
					k, v = item.split('=')
					if k in cat_dict.keys():
						sblist.append('{}: {}'.format(k, cat_dict[k][int(v)]))
			# for dvar in rule.discrete_vars.values():
			# 	print(dvar.name)
			# 	if dvar.name in cat_dict:
			# 		val_lbl = cat_dict[dvar.name][int(dvar.value)]
			# 		print(val_lbl)
			# 		sblist.append('{}: {}'.format(dvar.name, val_lbl))
			for cvar in rule.continuous_vars.values():
				if cvar.name in cat_dict:
					val_lbl = cat_dict[cvar.name][int(cvar.ubound)]
					sblist.append('{}: {}'.format(cvar.name, val_lbl))
			subpop = ' AND '.join(sblist)
			fptr.write('{}\n'.format(subpop))
			fptr.write('-----------------------------------------------------------\n')
			# fptr.write('Threshold test: ' + filter_status + '\n')
			fptr.write('Quartile 1: {}\n'.format(rule.q1))
			fptr.write('Quartile 3: {}\n'.format(rule.q3))
			fptr.write('IQR: {}\n'.format(rule.q1 - rule.q1))
			fptr.write('Q3 + {} *IQR: {}\n'.format(3, rule.q3 + 3*(rule.q3 - rule.q1)))
			# fptr.write('Population Threshold: {}\n'.format(rule.target_threshold))
			fptr.write('===========================================================\n\n')



if __name__ == '__main__':
	with open('mergedlst.pkl', 'rb') as f:
		mergedlist = pickle.load(f)
	
	generate_report('results.txt', mergedlist)