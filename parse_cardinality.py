import json

def parse_cardinality(fname):
    with open(fname) as f:
        cdict = json.load(f)

    res = cdict
    return res
