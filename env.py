# Query Optimization Environment


import os
import time
import psycopg2
from collections import defaultdict


# Parse analysis strings into cost and cardinality
def parse_perf(analysis):
    separators = ["cost=", "..", " rows=", " width=", ")"]
    cost, card = [0, 0], 0
    for row in analysis:
        line = row[0]
        if all(sep in line for sep in separators):
            idx = [line.find(sep) for sep in separators]
            numbers = [eval(line[idx[i]+len(separators[i]):idx[i+1]]) for i in range(4)]
            cost[0] += numbers[0]
            cost[1] += numbers[1]
            card += numbers[2] * numbers[3]
    return cost, card


class Query(object):
    def __init__(self, db_name="imdb", query_list="job_list.txt", query_dir="./job"):
        # Load queries from job benchmark
        assert os.path.isdir(query_dir), f"Please put joint order benchmark (job) in {query_dir}"
        with open(query_list, "r") as f:
            sql_file_list = [x.rstrip() for x in f.readlines()]

        self.query = dict()
        for sql_file in sql_file_list:
            fpath = os.path.join(query_dir, sql_file)
            assert os.path.isfile(fpath), f"{fpath} doesn't exist!"
            with open(fpath, "r") as f:
                self.query[sql_file] = f.readline().rstrip()

        # Connect to postgres
        self.conn = psycopg2.connect(dbname="imdb")
        self.cur = self.conn.cursor()

    def run(self):
        # Clear cache
        self.cur.execute("COMMIT;")
        self.cur.execute("DISCARD ALL;")

        # Run queries and evaluate performance
        perf = defaultdict(lambda: defaultdict(dict))
        for k, v in self.query.items():
            begin_time = time.time()
            self.cur.execute(v)
            perf[k]["time"] = time.time() - begin_time

            # TODO: check if cost and cardinality are accurate
            self.cur.execute(f"EXPLAIN {v}")
            perf[k]["cost"], perf[k]["card"] = parse_perf(self.cur.fetchall())
        return perf

    def __del__(self):
        self.conn.close()


if __name__ == "__main__":
    env = Query()
    perf = env.run()
    for k, v in perf.items():
        print(f"{k}: time={v['time']}, cost={v['cost']}, card={v['card']}")
