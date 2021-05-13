import os
import re
import sqlparse


def parse_sql(filename):
	# Open and read the file as a single buffer
	with open(filename, 'r') as f:
		predicates = [x.strip("\n,; ") for x in f.readlines()]
	while not predicates[0].startswith('FROM'):
		predicates.pop(0)
	from_info = []
	while not predicates[0].startswith('WHERE'):
		from_info.append(predicates.pop(0))
	predicates[0] = predicates[0].lstrip("WHERE ")

	tables = {data.lstrip("FROM ").split('AS')[1].strip(): data.lstrip("FROM ").split('AS')[0].strip() for data in from_info}

	predicates = list(filter(lambda x: x.count(".") >= 2 and "=" in x, predicates))
	predicates = list(map(lambda x: x.lstrip("AND ").split(" = "), predicates))

	return tables, predicates


def real_colomn(sname, tables):
	t, c = sname.split('.')
	return f"{tables[t]}.{c}"


if __name__ == '__main__':
	job_dir = 'job_formatted'
	job_names = os.listdir(job_dir)
	all_tables, all_colomns = [], []
	for job in job_names:
		tables, predicates = parse_sql(os.path.join(job_dir, job))
		all_tables += list(tables.values())
		for predicate in predicates:
			for sname in predicate:
				all_colomns.append(real_colomn(sname, tables))
	all_tables, all_colomns = list(set(all_tables)), list(set(all_colomns))
	with open("table_names.txt", "w") as f:
		f.write("\n".join(all_tables))
	with open("colomn_names.txt", "w") as f:
		f.write("\n".join(all_colomns))
