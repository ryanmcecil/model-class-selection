########################################
# Class to parse envrinoment variables.
########################################
import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv('.env', override=True)

@dataclass
class Env:
    data_dir: str = os.getenv("DATA_DIR", "default")
    plot_dir: str = os.getenv("PLOT_DIR", "default")
    table_dir: str = os.getenv("TABLE_DIR", "default")
    datamat_dir: str = os.getenv("DATAMAT_DIR", "default")
    num_cores: int = int(os.getenv("NUM_CORES", 1))

    def __post_init__(self):
        print('WARNING: ENV Variables')
        print('-------------------------------')
        print(f'Data directory path: {self.data_dir}')
        print(f'Plot directory path: {self.plot_dir}')
        print(f'Data Matrix directory: {self.datamat_dir}')
        print(f'Number of cores: {self.num_cores}')
        print('-------------------------------')
        if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
        if not os.path.exists(self.plot_dir):
                os.makedirs(self.plot_dir)