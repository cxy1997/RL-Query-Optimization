import os
import numpy as np
import sys
sys.path.insert(0, "..")
from parse_sql import parse_sql
from parse_cardinality import parse_cardinality


class JOB_env(object):
    def __init__(self,
                 job_list_file="../job_list.txt",
                 query_dir="../job_formatted",
                 cardinality_dir="../job_data",
                 table_names_file="../table_names.txt",
                 column_names_file="../column_names.txt"):
        with open(job_list_file) as f:
            self.job_list = [x.strip().rstrip(".sql") for x in f.readlines()]
        self.queries = [parse_sql(os.path.join(query_dir, f'{fname}.sql')) for fname in self.job_list]
        self.cardinalities = [parse_cardinality(os.path.join(cardinality_dir, f'{fname}.json')) for fname in self.job_list]
        with open(table_names_file) as f:
            self.all_tables = [x.strip() for x in f.readlines()]
        with open(column_names_file) as f:
            self.all_columns = [x.strip() for x in f.readlines()]
        self.logger = None

    def set_logger(self, logger):
        self.logger = logger

    def reset(self):
        self.index = np.random.randint(len(self.job_list))
        if self.logger is not None:
            self.logger.log(f"Restart from {self.job_list[self.index]}.")

        self.table_mapping, self.predicates = self.queries[self.index]
        self.tables = list(self.table_mapping.keys())

        self.state = {"tables": {table: cardinality},
                      "possible_actions": {action: predicates}}
        return self.state, self.get_info()

    def step(self, action):
        return self.state, reward, done, self.get_info()

    def get_info(self):
        return None


if __name__ == "__main__":
    env = JOB_env()
    print(env.job_list[0])
    print(env.queries[0])
    print(env.cardinalities[0])
