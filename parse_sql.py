import os
import re
import sqlparse

_DEBUG_ = True

def parse_sql(filename):
	# Open and read the file as a single buffer
	with open(filename, 'r') as f:
		predicates = [x.strip("\n,; ") for x in f.readlines()]
	while not predicates[0].startswith('FROM'):
		predicates.pop(0)
	from_info = []
	while not predicates[0].startswith('WHERE'):
		from_info.append(predicates.pop(0))
	predicates[0] = predicates[0][6:]

	tables = [data.split('AS')[1].strip() for data in from_info]
	if _DEBUG_:
		print('------------------')
		for x in from_info:
			print(x)
		print('------------------')
		print("tables:", tables)

	predicates = list(filter(lambda x: x.count(".") >= 2, predicates))
	predicates = list(map(lambda x: x.lstrip("AND "), predicates))

	if _DEBUG_:
		print('------------------')
		for x in predicates:
			print(x)

	# predicates = re.split('OR', condition_info, flags=re.IGNORECASE)
	# predicates = list(map(lambda x: re.split('AND', x, flags=re.IGNORECASE), predicates))

	# Parse the dataset conditions:
	# query_cart_pairs = []
	# query_filter_cmds  = []
	# conditions = re.split('AND', condition_info, flags=re.IGNORECASE)
	# for cond in conditions:
	# 	cond = cond.strip()
	# 	if '=' in cond:
	# 		conds = cond.split('=')
	# 		if _DEBUG_:
	# 			print("predicate", cond, cond.count('.') == 2)
	# 		if len(conds) == 2:
	# 			left_cond, right_cond = cond.split('=')
	# 			left_cond, right_cond = left_cond.strip(), right_cond.strip()
	# 			if (left_cond.split('.')[0] in tables) and (right_cond.split('.')[0] in tables):
	# 				query_cart_pairs.append(({left_cond.split('.')[0]: left_cond.split('.')[1]},
	# 					{right_cond.split('.')[0]: right_cond.split('.')[1]}))
	# 			else:
	# 				query_filter_cmds.append(cond)
	# 		else:
	# 				query_filter_cmds.append(cond)
	# 	else:
	# 		query_filter_cmds.append(cond)
	#
	#
	# # This is for debug output
	# if _DEBUG_:
	# 	print(query_cart_pairs)
	# 	print(query_filter_cmds)

	return tables, predicates

if __name__ == '__main__':
	job_dir = 'job_formatted'
	job_names = os.listdir(job_dir)[:10]
	for job in job_names:
		tables, predicates = parse_sql(os.path.join(job_dir, job))
		# print(tables)
		# print(predicates)
