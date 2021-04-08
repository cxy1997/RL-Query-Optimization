# RL Query Optimization

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
3. Start the server
    ```
    initdb $HOME/bin/pgsql/cluster0
    pg_ctl -D $HOME/bin/pgsql/cluster0 -l logfile start
    ```
