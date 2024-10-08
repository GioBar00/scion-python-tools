import enum
import os.path
from pathlib import Path
import pandas as pd
import shutil
import swifter
import numpy as np

OVERRIDE = True


class ActionType(enum.Enum):
    Received = 0
    ReceivedBcn = 1
    Propagated = 2
    PropagatedBcn = 3
    Originated = 4
    OriginatedBcn = 5
    Retrieved = 6
    Written = 7
    Processed = 8
    Executed = 9
    Completed = 10
    Algorithm = 11


class DeviceType(enum.Enum):
    CS = "cs"
    BR = "br"
    SD = "sd"
    RAC = "rac"


def to_device_type(s: str):
    if s.startswith("cs"):
        return DeviceType.CS
    if s.startswith("br"):
        return DeviceType.BR
    if s.startswith("sd"):
        return DeviceType.SD
    if s.startswith("rac"):
        return DeviceType.RAC
    raise ValueError(f"Unknown device type: {s}")


def load_df(devs_dir: str, devs: list[str]) -> pd.DataFrame:
    dtype = {
        'Type': str,
        'ID': str,
        'Next ID': str,
        'Data': float,
        'Time': str,
        'Size': int,
    }
    for i in range(7):
        dtype[f'Duration {i}'] = float

    dfs = [pd.read_csv(os.path.join(devs_dir, f"{dev}.csv"), sep=";", dtype=dtype) for dev in devs]
    for i in range(len(dfs)):
        dfs[i]["Device"] = devs[i]
    res = pd.concat(dfs)
    del dfs
    res.reset_index(drop=True, inplace=True)
    res['Time'] = pd.to_datetime(res['Time'])
    res['Type'] = res['Type'].swifter.apply(lambda x: ActionType[x])

    return res


def init_data_dir(name):
    sub_data_dir = os.path.join(data_dir, name)
    if not os.path.exists(sub_data_dir):
        os.makedirs(sub_data_dir)
    elif not os.path.isdir(sub_data_dir):
        raise ValueError('File exists with the same name as the scion directory')
    elif OVERRIDE:
        # Delete directory even if it is not empty
        shutil.rmtree(sub_data_dir)
        os.makedirs(sub_data_dir)
    elif len(os.listdir(sub_data_dir)) > 0:
        raise ValueError('Directory already exists')

    return sub_data_dir


def scion_to_data(scion_name):
    scion_dir = os.path.join(topology_dir, scion_name)
    scion_data_dir = init_data_dir(scion_name)

    scion_cses = [Path(f).stem for f in os.listdir(scion_dir) if f.endswith(".csv")]
    scion_df = load_df(scion_dir, scion_cses)

    # Origination
    scion_originated = scion_df[scion_df['Type'] == ActionType.Originated]
    scion_originated = scion_originated[scion_originated['Size'] == 4]
    scion_originated = scion_originated[['Duration 0', 'Duration 1', 'Duration 2', 'Duration 3']]
    scion_originated['Total'] = scion_originated.swifter.apply(
        lambda x: x['Duration 0'] + x['Duration 1'] + x['Duration 2'] + x['Duration 3'], axis=1)
    scion_originated = scion_originated[['Total']]
    scion_originated.to_csv(os.path.join(scion_data_dir, 'originated.csv'), index=False)
    del scion_originated

    # Propagation
    # Load retrieved
    scion_retrieved = scion_df[scion_df['Type'] == ActionType.Retrieved]
    scion_retrieved = scion_retrieved[scion_retrieved['Size'] == 2]
    scion_retrieved = scion_retrieved[['ID', 'Time', 'Data', 'Duration 0', 'Duration 1']]
    scion_retrieved['Total'] = scion_retrieved.swifter.apply(lambda x: x['Duration 0'] + x['Duration 1'], axis=1)
    scion_retrieved = scion_retrieved[['ID', 'Time', 'Data', 'Total']]

    # Load propagated
    scion_propagated = scion_df[scion_df['Type'] == ActionType.Propagated]
    scion_propagated = scion_propagated[scion_propagated['Size'] == 1]
    scion_propagated = scion_propagated[['ID', 'Time', 'Data', 'Duration 0']]
    scion_propagated = scion_propagated.rename(columns={'Duration 0': 'Total'})
    # rename id column
    scion_propagated = scion_propagated.rename(columns={'ID': 'FullID'})
    scion_propagated['ID'] = scion_propagated['FullID'].swifter.apply(lambda x: x.split(' ')[0])
    # merge retrieved and propagated
    scion_propagated = scion_propagated.merge(scion_retrieved, on='ID', how='left', suffixes=('', '_ret'))
    scion_propagated['Total'] = scion_propagated.swifter.apply(
        lambda x: x['Total'] + x['Total_ret'] / x['Data_ret'] * x['Data'], axis=1)
    scion_propagated = scion_propagated[['FullID', 'Time', 'Data', 'Total']]
    scion_propagated = scion_propagated.rename(columns={'FullID': 'ID'})
    del scion_retrieved

    # Load propagated beacons
    scion_propagated_bcn = scion_df[scion_df['Type'] == ActionType.PropagatedBcn]
    scion_propagated_bcn = scion_propagated_bcn[scion_propagated_bcn['Size'] == 3]
    scion_propagated_bcn = scion_propagated_bcn[['ID', 'Time', 'Data' 'Duration 0', 'Duration 1', 'Duration 2']]
    scion_propagated_bcn['Total'] = scion_propagated_bcn.swifter.apply(
        lambda x: x['Duration 0'] + x['Duration 1'] + x['Duration 2'], axis=1)
    scion_propagated_bcn = scion_propagated_bcn[['ID', 'Time', 'Data', 'Total']]
    scion_propagated_bcn = scion_propagated_bcn.rename(columns={'ID': 'FullID', 'Data': 'ID'})
    scion_propagated_bcn = scion_propagated_bcn.merge(scion_propagated, on='ID', how='left', suffixes=('', '_prop'))
    scion_propagated_bcn['Total'] = scion_propagated_bcn.swifter.apply(
        lambda x: x['Total'] + x['Total_prop'] / x['Data_prop'], axis=1)
    scion_propagated_bcn = scion_propagated_bcn[['Total']]
    scion_propagated_bcn.to_csv(os.path.join(scion_data_dir, 'propagated_bcn.csv'), index=False)
    del scion_propagated, scion_propagated_bcn


def irec_to_data(irec_name):
    irec_dir = os.path.join(topology_dir, irec_name)
    irec_data_dir = init_data_dir(irec_name)

    irec_cses = [Path(f).stem for f in os.listdir(irec_dir) if f.endswith(".csv")]
    irec_df = load_df(irec_dir, irec_cses)
    # TODO: Implement this


if __name__ == '__main__':
    topology_dir = os.path.join(os.path.dirname(__file__), 'tier1')
    data_dir = os.path.join(topology_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    scion_dir_names = []
    irec_dir_names = []
    # Load all folders in the topology directory
    for dir in os.listdir(topology_dir):
        if not os.path.isdir(os.path.join(topology_dir, dir)):
            continue
        if dir.startswith('scion'):
            scion_dir_names.append(dir)
        elif dir.startswith('irec'):
            irec_dir_names.append(dir)

    for scion_name in scion_dir_names:
        scion_to_data(scion_name)

    pass
