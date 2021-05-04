VALUE_DICT = {
    'seq_page_cost': [.25, .5, 1.0, 2.0, 4.0],
    'random_page_cost': [1.0, 2.0, 4.0, 8.0, 16.0],
    'effective_cache_size': ['64MB', '128MB', '256MB', '512MB', '1GB'],
    'cpu_tuple_cost': [.0025, .005, .01, .02, .04],
    'geqo_effort': [1, 3, 5, 7, 10],
    'enable_hashjoin': [True, False],
    'enable_mergejoin': [True, False],
    'enable_nestloop': [True, False],
    'enable_material': [True, False]
}

DEFAULTS_DICT = {
    'seq_page_cost': 2,
    'random_page_cost': 2,
    'effective_cache_size': 1,
    'cpu_tuple_cost': 2,
    'geqo_effort': 2,
    'enable_hashjoin': 0,
    'enable_mergejoin': 0,
    'enable_nestloop': 0,
    'enable_material': 0
}

def checkValues(values){
    for k in values:
        if values[k] >= len(VALUE_DICT[k]) or values[k] < 0:
            return False
        else
            return True
}


def setFlags(cur, values):
    '''
    cur is the cursor object associated with this connection

    values should be a dict in the same style as DEFAULT_DICT where each
    k, v pair represents setting attribute k to the v'th value in VALUE_DICT
    '''
    if not checkValues(values):
        raise ValueError(values)
    for k in VALUE_DICT:
        if k in values:
            command = 'set ' + k + " TO " + str(VALUE_DICT[k][values[k]]) + ";"
            cur.execute(command)
        else:
            command = 'set ' + k + " TO " + str(VALUE_DICT[k][DEFAULTS_DICT[k]]) + ";"
            cur.execute(command)

def setFlags(cur):
    setFlags(cur, {})

def freshDefaultDict():
    return DEFAULTS_DICT.copy

