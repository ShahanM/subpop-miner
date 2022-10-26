def delta_prune_empty(candidates, prune_delta, debug=False):
    # genlist = [{'rule_dict': {}, 'rule_str': '', 'support': 0, 'confidence': 0, \
        # 'q1': 0, 'q3': 0, 'threshold': 0}]
    pruned = [] 
    finallist = []
    for failitem in candidates:
        subrule = overlap(failitem, finallist)
        # if subrule:
        for sr in subrule:
            if abs(sr['threshold'] - failitem['threshold']) < prune_delta:
                # finallist.append(failitem)
                pruned.append(failitem)
                break
            # else:
        else:
            finallist.append(failitem)
    if debug:
        return {
            'cont_candidates': finallist,
            'pruned_rules': pruned
        }	
    return finallist


def prune_rules(rules_, debug=False):
    rules_.sort(key=lambda x: len(x['rule_dict'].keys()))
    prunedlist = []
    seen = []
    skipped = []
    for ruleset in rules_:
        rule_ = set(map(str.strip, ruleset['rule_str'].split('&')))
        if not skiprule(rule_, seen):
            seen.append(rule_)
            prunedlist.append(ruleset)
        else:
            skipped.append(rule_)
    if debug:
        return {
            'seen': seen,
            'skipped': skipped,
            'pruned': prunedlist
        }
    return prunedlist



def skiprule(rset, flst):
	if len(rset) <= 1:
		return False
	for frule in flst:
		if frule.issubset(rset):
			return True
	else:
		return False


def overlap(rset, setlst):
    rset_ = set(map(str.strip, rset['rule_str'].split('&')))
    overlappingrules = []
    for frule in setlst:
        fset = set(map(str.strip, frule['rule_str'].split('&')))
        if fset.issubset(rset_):
            overlappingrules.append(frule)
            # return frule
    # else:
        # return None
    return overlappingrules


def delta_prune(finalcontrol, candidates, prune_delta, debug=False):
    genlist = [{'rule_dict': {}, 'rule_str': '', 'support': 0, 'confidence': 0, \
        'q1': 0, 'q3': 0, 'threshold': 0}]
    pruned = []
    for failitem in candidates:
        subrule = overlap(failitem, finalcontrol)
        # if subrule:
        for sr in subrule:
            if abs(sr['threshold'] - failitem['threshold']) < prune_delta:
                pruned.append(failitem)
                break
                # genlist.append(failitem)
            # else:
        else:
            genlist.append(failitem)
    if debug:
        return {
            'cont_candidates': genlist,
            'pruned_rules': pruned
        }	
    return genlist
