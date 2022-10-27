from typing import List

from .rule import Rule


def delta_prune_empty(candidates: List[Rule], prune_delta: int)\
    -> List[Rule]:
    # genlist = [{'rule_dict': {}, 'rule_str': '', 'support': 0, 'confidence': 0, \
        # 'q1': 0, 'q3': 0, 'threshold': 0}]
    pruned = []
    finallist = []
    for candidate in candidates:
        subrules = overlap(candidate, finallist)
        # if subrule:
        for subrule in subrules:
            if abs(subrule.target_threshold - \
                candidate.target_threshold) < prune_delta:
                # finallist.append(failitem)
                pruned.append(candidate)
                break
            # else:
        else:
            finallist.append(candidate)
    # if debug:
    #     return {
    #         'cont_candidates': finallist,
    #         'pruned_rules': pruned
    #     }	
    return finallist


def prune_rules(rules: List[Rule]) -> List[Rule]:
    rules.sort(key=lambda rule: len(rule.itemset))
    # rules.sort(key=lambda x: len(x['rule_dict'].keys()))
    prunedlist = []
    seen = []
    for rule in rules:
        # rule = set(map(str.strip, ruleset['rule_str'].split('&')))
        rset = set(rule.itemset)
        if not skiprule(rset, seen):
            seen.append(rset)
            prunedlist.append(rule)
    return prunedlist


def skiprule(rset: set, flst: List[set]) -> bool:
	if len(rset) <= 1:
		return False
	for frule in flst:
		if frule.issubset(rset):
			return True
	else:
		return False


def overlap(rule: Rule, testpool: List[Rule]) -> List[Rule]:
    rset_ = set(rule.itemset)
    overlappingrules = []
    for frule in testpool:
        fset = set(frule.itemset)
        if fset.issubset(rset_):
            overlappingrules.append(frule)
    return overlappingrules

# def overlap(rset, setlst):
#     rset_ = set(map(str.strip, rset['rule_str'].split('&')))
#     overlappingrules = []
#     for frule in setlst:
#         fset = set(map(str.strip, frule['rule_str'].split('&')))
#         if fset.issubset(rset_):
#             overlappingrules.append(frule)
#             # return frule
#     # else:
#         # return None
#     return overlappingrules


def delta_prune(finalcontrol: List[Rule], candidates: List[Rule], \
    prune_delta: int) -> List[Rule]:
    # genlist = [{'rule_dict': {}, 'rule_str': '', 'support': 0, 'confidence': 0, \
        # 'q1': 0, 'q3': 0, 'threshold': 0}]
    # pruned = []
    outlist = []
    for candidate in candidates:
        subrules = overlap(candidate, finalcontrol)
        # if subrule:
        pruned = filter(lambda subrule: abs(subrule.target_threshold - \
            candidate.target_threshold) < prune_delta, subrules)
        if len(list(pruned)) == 0:
            outlist.append(candidate)

        # for subrule in subrules:
            # if abs(subrule.target_threshold - candidate.target_threshold) < prune_delta:
                # pruned.append(candidate)
                # break
                # genlist.append(failitem)
            # else:
        # else:
            # genlist.append(failitem)
    return outlist


def write_rules_file(ruleslist, fname, popthresh):
    with open(fname, 'w') as fptr:
        fptr.write('Total Rules: {}'.format(len(ruleslist)))
        fptr.write('\n\n')
        for rule_ in ruleslist:
            sblist = []
            skip = False
            for k, v in rule_['rule_dict'].items():
                if k in cat_dict.keys():
                    if type(v) == dict:
                        if int(v['ubound']) <= 3:
                            print(rule_)
                            skip = True
                        if int(v['ubound']) == 4 and len(rule_['rule_dict'].keys()) > 1:
                            skip = True
                        sblist.append('{}: {}'.format(k, cat_dict[k][int(v['ubound'])]))
                    else:
                        sblist.append('{}: {}'.format(k, cat_dict[k][v]))
            subpop = ' AND '.join(sblist)
            # subpop = ' AND '.join(['{}: {}'.format(k, cat_dict[k][v]) \
            #     for k, v in rule_['rule_dict'].items() if k in cat_dict.keys()])
            q1 = rule_['q1']
            q3 = rule_['q3']
            mu=3
            threshold = rule_['threshold']

            conf = rule_['confidence']
            supp = rule_['support']
            rulestr = rule_['rule_str']
            
            filter_status = ''
            if rule_['threshold'] <= popthresh:
                filter_status = 'PASS'
            else:
                filter_status = 'FAIL'
                continue

            if skip:
                continue

            pretty_print_to_file(rule_string=rulestr, confidence=conf, \
                support=supp, subpop=subpop, filter_status=filter_status, \
                q1=q1, q3=q3, mu=mu, threshold=threshold, fileptr=fptr)


def pretty_print_to_file(rule_string, confidence, support, subpop, \
        filter_status, q1, q3, mu, threshold, fileptr):	
    fileptr.write('\nShowing for subpopulation {}\n'.format(rule_string))
    fileptr.write('Confidence {}\n'.format(confidence))
    fileptr.write('Support {}\n'.format(support))
    fileptr.write('{}\n'.format(subpop))
    fileptr.write('-----------------------------------------------------------\n')
    fileptr.write('Threshold test: ' + filter_status + '\n')
    fileptr.write('Quartile 1: {}\n'.format(q1))
    fileptr.write('Quartile 3: {}\n'.format(q3))
    fileptr.write('IQR: {}\n'.format(q3 - q1))
    fileptr.write('Q3 + {} *IQR: {}\n'.format(mu, threshold))
    fileptr.write('Population Threshold: {}\n'.format(pop_threshold))
    fileptr.write('===========================================================\n\n')
