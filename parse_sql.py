import os
import re

_DEBUG_ = False

def parse_sql(filename):
	# Open and read the file as a single buffer
	fd = open(filename, 'r')
	sqlFile = fd.read()
	fd.close()
	# The sql files are written in a single line	
	sqlFile = sqlFile.split(';')[0]
	# Sperate `FROM` first
	select_info, from_info = sqlFile.split('FROM')
	# Sperate `WHERE` second
	from_info, condition_info = from_info.split('WHERE')

	# This is for debug output
	if _DEBUG_:
		print('------------------')
		print(select_info)
		print('------------------')
		print(from_info)
		print('------------------')
		print(condition_info)
		print('------------------')

	# Determine how many datasets used:
	# data_map is a dict where key is abbrevation, and value is full name
	data_map = {}
	datasets = from_info.split(',')
	for data in datasets:
		full_name, short_name = data.split('AS')
		full_name, short_name = full_name.strip(), short_name.strip()
		data_map[short_name] = full_name
	# This is for debug output
	if _DEBUG_:
		print(data_map)	

	# Parse the dataset conditions:
	query_cart_pairs = []
	query_filter_cmds  = []
	conditions = re.split('AND', condition_info, flags=re.IGNORECASE)
	for cond in conditions:
		cond = cond.strip()
		if '=' in cond:
			conds = cond.split('=')
			if len(conds) == 2:
				left_cond, right_cond = cond.split('=')
				left_cond, right_cond = left_cond.strip(), right_cond.strip()
				if (left_cond.split('.')[0] in data_map.keys()) and (right_cond.split('.')[0] in data_map.keys()):
					query_cart_pairs.append(({left_cond.split('.')[0]: left_cond.split('.')[1]},
						{right_cond.split('.')[0]: right_cond.split('.')[1]}))
				else:
					query_filter_cmds.append(cond)
			else:
					query_filter_cmds.append(cond)
		else:
			query_filter_cmds.append(cond)	


	# This is for debug output
	if _DEBUG_:
		print(query_cart_pairs)
		print(query_filter_cmds)

	return data_map, query_cart_pairs, query_filter_cmds

if __name__ == '__main__':
	job_dir = 'job'
	job_names = os.listdir(job_dir)[:1]
	for job in job_names:
		parse_sql(os.path.join(job_dir, job))
