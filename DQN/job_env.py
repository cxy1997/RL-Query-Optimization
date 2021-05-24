import os
import numpy as np
import sys
sys.path.append('../')
from parse_sql import parse_sql
from parse_cardinality import parse_cardinality
from copy import deepcopy


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
        self.all_cardinalities = [parse_cardinality(os.path.join(cardinality_dir, f'{fname}.json')) for fname in self.job_list]
        with open(table_names_file) as f:
            self.all_tables = [x.strip() for x in f.readlines()]
        with open(column_names_file) as f:
            self.all_columns = [x.strip() for x in f.readlines()]
        self.logger = None

    def __len__(self):
        return len(self.job_list)

    def set_logger(self, logger):
        self.logger = logger

    def reset(self, index=None):
        if index is not None:
            self.index = index
        else:
            self.index = np.random.randint(len(self.job_list))

        if self.logger is not None:
            self.logger.log(f"Restart from {self.job_list[self.index]}.")
        else:
            print(f"Restart from {self.job_list[self.index]}.")

        self.table_mapping, self.predicates = self.queries[self.index]
        self.tables = list(self.table_mapping.keys())  # [an,cc,cct1]
        self.cardinalities = self.all_cardinalities[self.index]

        self.state = {}
        self.state["tables"] = [(self.get_onehot_encoding(item["basetable"], self.all_tables),
                                item["cardinality"]) for item in self.cardinalities["relations"]]
        
        self.state["possible_actions"] = {}
        for predicate in self.predicates:
            entry_0, entry_1 = predicate
            idx_0, idx_1 = self.tables.index(entry_0.split('.')[0]), self.tables.index(entry_1.split('.')[0])
            value_0 = self.get_onehot_encoding(self.table_mapping[entry_0.split('.')[0]] + '.' + entry_0.split('.')[1], self.all_columns)
            value_1 = self.get_onehot_encoding(self.table_mapping[entry_1.split('.')[0]] + '.' + entry_1.split('.')[1], self.all_columns)
            
            # make sure idx_0 < idx_1
            if idx_0 < idx_1:
                self.state["possible_actions"][(idx_0, idx_1)] = (value_0, value_1)
            else:
                self.state["possible_actions"][(idx_1, idx_0)] = (value_0, value_1)

        # This is an example.
        # noted that encoding is np.array instead of list

        # [([1, 0, 0], 20), ([0, 1, 0], 30), ([0, 0, 1], 10)]
        # {(0, 1): (21, 25), (1, 2): (12, 21)}

        # take action (0, 1)
        # [([1, 1, 0], 5), ([0, 0, 1], 10)]
        # {(0, 1): (12, 21)}

        # self.state = {"tables": [(id, cardinality), ...]},
        #               "possible_actions": {action: predicate}}

        # Finally double check if the order of name in self.table_mapping and self.cardinalities["relations"] is same
        # for (key_1, key_2) in zip(self.table_mapping.keys(), self.cardinalities["relations"]):
        #     assert key_1 == key_2["name"]

        # double check number of possible_actions matches number of env.cardinalities['joins']
        assert len(list(self.state['possible_actions'].keys())) == len(self.cardinalities['joins'])

        # map encoding to cardinalities, from env.cardinalities['sizes']
        self.e2c = {}

        for temp in self.cardinalities['sizes']:
            e = np.zeros(len(self.all_tables),dtype=int)
            for temp_table in temp['relations']:
                e += self.get_onehot_encoding(self.table_mapping[temp_table], self.all_tables)
            c = temp['cardinality']
            self.e2c[e.tobytes()] = c

        return deepcopy(self.state), self.get_info()

    def get_greedy_action(self):
        actions = {}
        for action in self.state["possible_actions"].keys():
            actions[action] = self.e2c[(self.state['tables'][action[0]][0] + self.state['tables'][action[1]][0]).tobytes()]
        print(sorted(actions.values()))
        return sorted(actions.keys(), key=actions.get)[0]

    def step(self, action):

        # here action is the tuple of the actual merged two table indexes
        (idx_0, idx_1) = action
        assert idx_0 < idx_1
        
        table_1 = self.state['tables'].pop(idx_1)
        table_0 = self.state['tables'].pop(idx_0)
        join_e = table_0[0] + table_1[0]
        join_c = self.e2c[join_e.tobytes()]
        table_join = (join_e, join_c)
        self.state['tables'].append(table_join)

        done = (len(self.state['tables'])==1)
        # currently set the reward function as -log(c)
        reward = {
            "direct_c": -join_c,
            "log_c": np.clip(-np.log(join_c + 1) / 10, -10, 10),
            "log_reduced_c": np.clip(np.log(max(table_0[1] - join_c, table_1[1] - join_c,  1)) / 10, -10, 10),
            "log_scale": np.clip(np.log((table_0[1]+1) * (table_1[1]+1) / (join_c+1)) / 10, -10, 10)
        }

        if done:
            self.state['possible_actions'] = None
        
        else:

            new_action_dict = {}

            for temp_tuple in self.state['possible_actions']:
                if temp_tuple == (idx_0,idx_1):
                    continue
                new_action = [None,None]
                for i in range(2):
                    idx = temp_tuple[i]
                    if idx < idx_0:
                        new_action[i] = idx
                    elif idx == idx_0:
                        new_action[i] = len(self.state['tables'])-1
                    elif idx > idx_0:
                        if idx < idx_1:
                            new_action[i] = idx-1
                        elif idx == idx_1:
                            new_action[i] = len(self.state['tables'])-1
                        elif idx > idx_1:
                            new_action[i] = idx-2
                if new_action[0]<new_action[1]:
                    new_tuple = tuple(new_action)
                else:
                    new_tuple = (new_action[1],new_action[0])
                new_action_dict[new_tuple] = self.state['possible_actions'][temp_tuple]
            
            self.state['possible_actions'] = new_action_dict
        
        return deepcopy(self.state), reward, done, self.get_info()

    def get_info(self):
        return None

    def get_onehot_encoding(self, query, keys_list):
        return np.array([int(query == key) for key in keys_list])


if __name__ == "__main__":
    env = JOB_env()
    env.reset()
