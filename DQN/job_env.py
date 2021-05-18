import os
import numpy as np
import sys
sys.path.append('../')
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
        self.cardinalities = self.cardinalities[self.index]

        self.state = {}
        self.state["tables"] = [(self.get_onehot_encoding(item["basetable"], self.all_tables),
                                item["cardinality"]) for item in self.cardinalities["relations"]]
        # print('--------------------------------------')
        # print(self.state["tables"])
        self.state["possible_actions"]= {}
        predicates_name_list = [key for key in self.table_mapping.keys()]
        for predicate in self.predicates:
            entry_0, entry_1 = predicate
            idx_0, idx_1 = predicates_name_list.index(entry_0.split('.')[0]), predicates_name_list.index(entry_1.split('.')[0])
            value_0 = self.get_onehot_encoding(self.table_mapping[entry_0.split('.')[0]] + '.' + entry_0.split('.')[1], self.all_columns)
            value_1 = self.get_onehot_encoding(self.table_mapping[entry_1.split('.')[0]] + '.' + entry_1.split('.')[1], self.all_columns)
            self.state["possible_actions"][(idx_0, idx_1)] = (value_0, value_1)

        # print('--------------------------------------')
        # print(self.state["possible_actions"])

        # This is an example.
        # [([1, 0, 0], 20), ([0, 1, 0], 30), ([0, 0, 1], 10)]
        # {(0, 1): (21, 25), (1, 2): (12, 21)}

        # take action (0, 1)
        # [([1, 1, 0], 5), ([0, 0, 1], 10)]
        # {(0, 1): (12, 21)}

        # self.state = {"tables": [(id, cardinality), ...]},
        #               "possible_actions": {action: predicate}}
        # return self.state, self.get_info()

        # Finally double check if the order of name in self.table_mapping and self.cardinalities["relations"] is same
        for (key_1, key_2) in zip(self.table_mapping.keys(), self.cardinalities["relations"]):
            assert key_1 == key_2["name"]

        return self.state, self.get_info()

    def step(self, action):
        return self.state, reward, done, self.get_info()

    def get_info(self):
        return None

    def get_onehot_encoding(self, query, keys_list):
        return [int(query == key) for key in keys_list]


if __name__ == "__main__":
    env = JOB_env()
    # print('--------------------------------------')
    # print(env.job_list[0])
    # print('--------------------------------------')
    # print(env.queries[0])
    # print('--------------------------------------')
    # print(env.cardinalities[0])
    # print('--------------------------------------')
    env.reset()
