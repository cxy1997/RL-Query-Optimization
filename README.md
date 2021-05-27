# RL Query Optimization

## Training Deep-Q Network
```bash
cd DQN/
python train_dqn.py --save-path test-run --reward-mode [log_c/log_reduced_c/fusion]
```

## Installation

### Install Java

1. Download Java 10.0.2 from [google drive](https://drive.google.com/drive/folders/1VR58_7_6ZVpVw_DaSdNxrD8TAd6hCiRg?usp=sharing), then unzip them with `tar -xf`.
2. Add these lines to `~/.bashrc`
    ```bash
    export JAVA_HOME=/path/to/jdk-10.0.2
    export PATH=$JAVA_HOME/bin:$PATH
    ```
3. `source ~/.bashrc`

### Install Calcite

```
git clone https://github.com/apache/calcite.git
cd calcite/example/csv
./sqlline
```

### Install PostgreSQL
1. Download, compile, and install
    ```bash
    wget https://ftp.postgresql.org/pub/source/v13.2/postgresql-13.2.tar.gz
    tar -xf postgresql-13.2.tar.gz
    cd postgresql-13.2
    ./configure --prefix=$HOME/bin/pgsql --with-python
    make -j8
    make install -j8
    ```
2. Add these lines to `~/.bashrc`
    ```bash
    export LD_LIBRARY_PATH=$HOME/bin/pgsql/lib:$LD_LIBRARY_PATH
    export PATH=$HOME/bin/pgsql/bin:$PATH
    export MANPATH=$HOME/bin/pgsql/share/man:$MANPATH
    ```
3. Select an unused port number, update it in [google sheets](https://docs.google.com/spreadsheets/d/1YjB8PJfFlHAyexqW7ha2_DQgH375APIZymi4AAbfr0U/edit?usp=sharinghttps://docs.google.com/spreadsheets/d/1YjB8PJfFlHAyexqW7ha2_DQgH375APIZymi4AAbfr0U/edit?usp=sharing), and add these lines to `~/.bashrc`
    ```bash
    export dbport=xxxx
    alias createdb='$HOME/bin/pgsql/bin/createdb -h localhost -p $dbport'
    alias psql='$HOME/bin/pgsql/bin/psql -h localhost -p $dbport'
    ```
4. Start the server
    ```
    source ~/.bashrc
    initdb $HOME/bin/pgsql/cluster0
    pg_ctl -D $HOME/bin/pgsql/cluster0 -o "-p $dbport" -l logfile start
    ```

### Prepare database and queries
1. Download IMDB data from [google drive](https://drive.google.com/file/d/1Qhk-Mf2Otc6C9e-vIm2OQyKoko7kK_--/view?usp=sharing). This is a big file, it would take a lot of storage.

2. Create database
    ```bash
    createdb imdb
    psql imdb -f imdb_dump.sql
    ```


### Install Python dependencies
 ```bash
 python -m pip install psycopg2
 ```

### Usage
1. Put `job` folder under `RL-Query-Optimization`

2. 
   ```bash
   python env.py
   ```
3. Record results in [this google sheet](https://docs.google.com/spreadsheets/d/1YjB8PJfFlHAyexqW7ha2_DQgH375APIZymi4AAbfr0U/edit?usp=sharing)

### Parse SQL files
1. Run the function of `parse_sql()` which takes the SQL filepath as the input in `parse_sql.py`

2. The output of `parse_sql()` should be in three parts:
* `dataset_map` which is a `dict` where the keys are in short name for the data used and the corresponding value is the data's full name. An example is here:

  ```
  {'cn': 'company_name', 'ct': 'company_type'}
  ```

  You can loop  `dataset_map.keys()`  to see how many datasets are used.

* `query_cart_pairs` which is a `list` contains tuples for the join query for two datasets. An example is here:

  ```
  [
  ({'mi': 'movie_id'}, {'t': 'id'}), 
  ({'it2': 'id'}, {'mi': 'info_type_id'})
  ]
  ```

  Inside each tuple, every element is a `dict` and the key is the short name for dataset and value is the attribute used.

*  `query_filter_cmds` which is a `list` contains other SQL executing commands. An example is:

   ```
   ["cn.country_code ='[us]'", "ct.kind ='production companies'"]
   ```

    

