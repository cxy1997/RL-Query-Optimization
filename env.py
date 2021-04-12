# Query Optimization Environment


import os
import time
import psycopg2
from collections import defaultdict
from tqdm import tqdm


def parse_time(line):
    separators = ["Execution Time: ", " ms"]
    assert all(sep in line for sep in separators)
    idx = [line.find(sep) for sep in separators]
    return eval(line[idx[0]+len(separators[0]):idx[1]])

def parse_cost(line):
    separators = ["cost=", "..", " rows="]
    assert all(sep in line for sep in separators)
    idx = [line.find(sep) for sep in separators]
    return [eval(line[idx[i]+len(separators[i]):idx[i+1]]) for i in range(2)]


def parse_rows_sum(analysis):
    lines = [row[0] for row in analysis if "->" in row[0] and "never executed" not in row[0]]
    for l in lines:
        print(l)
    process = lambda line: eval(line[line.rfind(" rows=") + 6:line.rfind(" loops=")])
    return sum(map(process, lines))


def parse_rows_upperbound(analysis):
    lines = [row[0] for row in analysis if "->" in row[0] and "never executed" not in row[0]]
    process = lambda line: [(line.find("->") + 4) // 6, eval(line[line.rfind(" rows=") + 6:line.rfind(" loops=")])]
    row_info = list(map(process, lines)) + [[0, 0]]
    upperbound, stack = 0, []
    for level, rows in row_info:
        if len(stack) > 0:
            for l in range(stack[-1][0], level, -1):
                base = 1
                while len(stack) > 0 and stack[-1][0] == l:
                    base *= stack.pop()[1]
                upperbound += base
        stack.append([level, rows])
    return upperbound


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
        self.conn = psycopg2.connect(dbname=db_name)
        self.cur = self.conn.cursor()

    def run(self):
        # Clear cache
        self.cur.execute("COMMIT;")
        self.cur.execute("DISCARD ALL;")
        self.cur.execute("SET max_parallel_workers_per_gather = 0;")

        # Run queries and evaluate performance
        perf = defaultdict(lambda: defaultdict(dict))
        for k, v in tqdm(self.query.items()):
            self.cur.execute(f"EXPLAIN ANALYZE VERBOSE {v}")
            analysis = self.cur.fetchall()
            perf[k]["cost"] = parse_cost(analysis[0][0])
            perf[k]["time"] = parse_time(analysis[-1][0])
            perf[k]["rows_sum"] = parse_rows_sum(analysis)
            perf[k]["rows_upperbound"] = parse_rows_upperbound(analysis)
        return perf

    def __del__(self):
        self.conn.close()


if __name__ == "__main__":
    env = Query()
    perf = env.run()

    total_time, total_min_cost, total_max_cost, total_rows_sum, total_rows_upperbound = 0, 0, 0, 0, 0
    s = []
    for k, v in perf.items():
        s.append(f"{k}\t{v['time']}\t{v['cost'][0]}\t{v['cost'][1]}\t{v['rows_sum']}\t{v['rows_upperbound']}")
        total_time += v['time']
        total_min_cost += v['cost'][0]
        total_max_cost += v['cost'][1]
        total_rows_sum += v['rows_sum']
        total_rows_upperbound += v['rows_upperbound']
    s = ["Query\ttime\tcost_min\tcost_max\trows_sum\trows_upperbound", f"Total\t{total_time}\t{total_min_cost}\t{total_max_cost}\t{total_rows_sum}\t{total_rows_upperbound}"] + s
    os.makedirs("logs", exist_ok=True)
    fname = "logs/postgres_query_optimizer3.txt"
    with open(fname, "w") as f:
        f.write("\n".join(s))
    print(fname)
