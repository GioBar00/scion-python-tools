import enum
import os.path
from pathlib import Path
import pandas as pd
import shutil
import swifter
import numpy as np
import humanfriendly

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
        'Data': str,
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
    #res['Type'] = res['Type'].swifter.apply(lambda x: ActionType[x])

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

    # Create subdirectories
    os.makedirs(os.path.join(sub_data_dir, 'fc'))
    os.makedirs(os.path.join(sub_data_dir, 'usage'))
    return sub_data_dir


def fc_to_data(source_dir, dest_dir):
    fc_devs = [Path(f).stem for f in os.listdir(source_dir) if f.endswith(".txt")]

    def filename_to_isd_as(filename):
        filename = filename.split('.')[0]
        for dev_type in DeviceType:
            if filename.startswith(dev_type.value):
                filename = filename[len(dev_type.value):]
                break
        isd_as = filename.split('_')[0]
        isd_as = isd_as + '-' + filename.replace(isd_as + '_', '').replace('_', ':')
        return isd_as

    def parse_fc(fc_file):
        with open(fc_file, 'r') as f:
            lines = f.readlines()
        # Remove empty lines
        lines = [line for line in lines if line.strip()]
        # Convert first line to datetime
        start = pd.to_datetime(lines[0].strip())
        end = pd.to_datetime(lines[-1].strip())
        res = {
            'Start': start,
            'End': end,
            'Others': {}
        }
        for line in lines[1:-1]:
            line = line.strip().split('_')
            other = line[0]
            time = pd.to_datetime(line[1])
            res['Others'][other] = time
        return res

    fcs = {}

    for dev in fc_devs:
        isd_as = filename_to_isd_as(dev)
        fc_file = os.path.join(source_dir, f"{dev}.txt")
        res = parse_fc(fc_file)
        fcs[isd_as] = res

    # Create dataframe
    df = pd.DataFrame(columns=['Isd-As1', 'Isd-As2', 'Duration'])
    for isd_as1, data in fcs.items():
        for isd_as2, time in data['Others'].items():
            duration = time - min(data['Start'], fcs[isd_as2]['Start'])
            df.loc[len(df)] = [isd_as1, isd_as2, duration.total_seconds()]
        # duration = data['End'] - data['Start']
        # df.loc[len(df)] = [isd_as1, isd_as1, duration.total_seconds()]

    df.to_csv(os.path.join(dest_dir, 'fc.csv'), index=False)


def usage_to_data(source_dir, dest_dir):
    cpu_file = os.path.join(source_dir, 'cpu.csv')
    if os.path.exists(cpu_file):
        shutil.copy(cpu_file, dest_dir)
    mem_file = os.path.join(source_dir, 'mem.csv')
    if os.path.exists(mem_file):
        shutil.copy(mem_file, dest_dir)
    net_rx_file = os.path.join(source_dir, 'net_rx.csv')
    if os.path.exists(net_rx_file):
        shutil.copy(net_rx_file, dest_dir)
    net_tx_file = os.path.join(source_dir, 'net_tx.csv')
    if os.path.exists(net_tx_file):
        shutil.copy(net_tx_file, dest_dir)

    cpu_df = pd.read_csv(cpu_file) if os.path.exists(cpu_file) else None
    mem_df = pd.read_csv(mem_file) if os.path.exists(mem_file) else None
    net_rx_df = pd.read_csv(net_rx_file) if os.path.exists(net_rx_file) else None
    net_tx_df = pd.read_csv(net_tx_file) if os.path.exists(net_tx_file) else None

    def process_usage_df(df: pd.DataFrame, parse_size=False, parse_speed=False):
        if df is None:
            return None
        # Delete Time column
        df = df.drop(columns=['Time'])

        # Convert all columns to numeric
        if parse_size:
            df = df.applymap(lambda x: humanfriendly.parse_size(x) if isinstance(x, str) else x)
        elif parse_speed:
            df = df.applymap(lambda x: humanfriendly.parse_size(x.replace('/s', '')) if isinstance(x, str) else x)

        dev_types_df = {}
        for dev_type in DeviceType:
            # Filter columns that start with the device type
            dev_cols = [col for col in df.columns if col.startswith("kathara-" + dev_type.value)]
            if len(dev_cols) == 0:
                continue
            # Rename columns with function that removes the device type prefix
            dev_types_df[dev_type] = df[dev_cols].rename(columns=lambda x: x.split('-')[1] + "-" + str.join(':', x.split('-')[2:-3]))

        return dev_types_df

    cpu_df = process_usage_df(cpu_df)
    mem_df = process_usage_df(mem_df, parse_size=True)
    net_rx_df = process_usage_df(net_rx_df, parse_speed=True)
    net_tx_df = process_usage_df(net_tx_df, parse_speed=True)
    # Compute the peak and median for each column and create DataFrames
    def create_peak_median_df(df_dict):
        if df_dict is None:
            return None
        data = []
        for dev_type, df in df_dict.items():
            peak = df.max(axis=0)
            median = df.median(axis=0)
            for col in df.columns:
                data.append([dev_type.value, peak[col], median[col]])
        return pd.DataFrame(data, columns=["Dev Type", "Peak", "Median"])

    cpu_peak_median_df = create_peak_median_df(cpu_df)
    mem_peak_median_df = create_peak_median_df(mem_df)
    net_rx_peak_median_df = create_peak_median_df(net_rx_df)
    net_tx_peak_median_df = create_peak_median_df(net_tx_df)

    if cpu_peak_median_df is not None:
        cpu_peak_median_df.to_csv(os.path.join(dest_dir, 'cpu-pm.csv'), index=False)
    if mem_peak_median_df is not None:
        mem_peak_median_df.to_csv(os.path.join(dest_dir, 'mem-pm.csv'), index=False)
    if net_rx_peak_median_df is not None:
        net_rx_peak_median_df.to_csv(os.path.join(dest_dir, 'net_rx-pm.csv'), index=False)
    if net_tx_peak_median_df is not None:
        net_tx_peak_median_df.to_csv(os.path.join(dest_dir, 'net_tx-pm.csv'), index=False)


def scion_to_data(scion_name):
    scion_dir = os.path.join(topology_dir, scion_name)
    scion_data_dir = init_data_dir(scion_name)

    scion_fc_dir = os.path.join(scion_dir, 'fc')
    if os.path.exists(scion_fc_dir):
        fc_to_data(scion_fc_dir, os.path.join(scion_data_dir, 'fc'))

    scion_usage_dir = os.path.join(scion_dir, 'usage')
    if os.path.exists(scion_usage_dir):
        usage_to_data(scion_usage_dir, os.path.join(scion_data_dir, 'usage'))

    scion_cses = [Path(f).stem for f in os.listdir(scion_dir) if f.endswith(".csv")]
    scion_df = load_df(scion_dir, scion_cses)
    # Handle Beacon: ActionType.ReceivedBcn
    scion_beacon = scion_df[scion_df['Type'] == ActionType.ReceivedBcn.name]
    scion_beacon = scion_beacon[scion_beacon['Size'] == 6]
    scion_beacon = scion_beacon[['Duration 0', 'Duration 1', 'Duration 2', 'Duration 3', 'Duration 4', 'Duration 5']]
    scion_beacon['Total'] = scion_beacon.swifter.apply(
        lambda x: x['Duration 0'] + x['Duration 1'] + x['Duration 2'] + x['Duration 3'] + x['Duration 4'] + x[
            'Duration 5'], axis=1)
    #scion_large = scion_beacon[scion_beacon['Total'] > 2]
    scion_beacon = scion_beacon[['Total']]
    scion_beacon.to_csv(os.path.join(scion_data_dir, 'handle_bcn.csv'), index=False)
    del scion_beacon

    # Propagation Filtering: ActionType.Propagated
    scion_propagation = scion_df[scion_df['Type'] == ActionType.Propagated.name]
    scion_propagation = scion_propagation[scion_propagation['Size'] == 2]
    # Data column as int
    scion_propagation['Data'] = scion_propagation['Data'].astype(int)
    scion_propagation = scion_propagation[['Duration 0', 'Duration 1', 'Data']]
    scion_propagation['Total'] = scion_propagation.swifter.apply(
        lambda x: (x['Duration 0'] + x['Duration 1']) / x['Data'], axis=1)
    scion_propagation = scion_propagation[['Total']]
    scion_propagation.to_csv(os.path.join(scion_data_dir, 'prop_filter.csv'), index=False)
    del scion_propagation

    # Propagation: ActionType.PropagatedBcn
    scion_propagation_bcn = scion_df[scion_df['Type'] == ActionType.PropagatedBcn.name]
    scion_propagation_bcn = scion_propagation_bcn[scion_propagation_bcn['Size'] == 4]
    scion_propagation_bcn = scion_propagation_bcn[['Duration 0', 'Duration 1', 'Duration 2', 'Duration 3']]
    # Maybe remove Duration 2 -> included in handle beacon
    scion_propagation_bcn['Total'] = scion_propagation_bcn.swifter.apply(
        #lambda x: x['Duration 0'] + x['Duration 1'] + x['Duration 2'] + x['Duration 3'], axis=1)
        lambda x: x['Duration 0'] + x['Duration 1'] + x['Duration 3'], axis = 1)
    scion_propagation_bcn = scion_propagation_bcn[['Total']]
    scion_propagation_bcn.to_csv(os.path.join(scion_data_dir, 'prop_bcn.csv'), index=False)
    del scion_propagation_bcn

    # Originate beacon: ActionType.OriginatedBcn
    scion_originated_bcn = scion_df[scion_df['Type'] == ActionType.OriginatedBcn.name]
    scion_originated_bcn = scion_originated_bcn[scion_originated_bcn['Size'] == 4]
    scion_originated_bcn = scion_originated_bcn[['Duration 0', 'Duration 1', 'Duration 2', 'Duration 3']]
    # Maybe remove Duration 2 -> included in handle beacon
    scion_originated_bcn['Total'] = scion_originated_bcn.swifter.apply(
        #lambda x: x['Duration 0'] + x['Duration 1'] + x['Duration 2'] + x['Duration 3'], axis=1)
        lambda x: x['Duration 0'] + x['Duration 1'] + x['Duration 3'], axis = 1)
    scion_originated_bcn = scion_originated_bcn[['Total']]
    scion_originated_bcn.to_csv(os.path.join(scion_data_dir, 'originated_bcn.csv'), index=False)
    del scion_originated_bcn

    # Written: ActionType.Written
    scion_written = scion_df[scion_df['Type'] == ActionType.Written.name]
    scion_written = scion_written[scion_written['Size'] == 2]
    # Data column as int
    scion_written['Data'] = scion_written['Data'].astype(int)
    scion_written = scion_written[['Duration 0', 'Duration 1', 'Data']]
    scion_written['Total'] = scion_written.swifter.apply(lambda x: (x['Duration 0'] + x['Duration 1']) / x['Data'], axis=1)
    scion_written = scion_written[['Total']]
    scion_written.to_csv(os.path.join(scion_data_dir, 'written.csv'), index=False)
    del scion_written

    # DONE
    del scion_df


def irec_to_data(irec_name):
    irec_dir = os.path.join(topology_dir, irec_name)
    irec_data_dir = init_data_dir(irec_name)

    irec_fc_dir = os.path.join(irec_dir, 'fc')
    if os.path.exists(irec_fc_dir):
        fc_to_data(irec_fc_dir, os.path.join(irec_data_dir, 'fc'))

    irec_usage_dir = os.path.join(irec_dir, 'usage')
    if os.path.exists(irec_usage_dir):
        usage_to_data(irec_usage_dir, os.path.join(irec_data_dir, 'usage'))


    irec_devs = [Path(f).stem for f in os.listdir(irec_dir) if f.endswith(".csv")]
    irec_df = load_df(irec_dir, irec_devs)
    # Handle Beacon: ActionType.ReceivedBcn
    irec_beacon = irec_df[irec_df['Type'] == ActionType.ReceivedBcn.name]
    irec_beacon = irec_beacon[irec_beacon['Size'] == 6]
    irec_beacon = irec_beacon[['Duration 0', 'Duration 1', 'Duration 2', 'Duration 3', 'Duration 4', 'Duration 5']]
    irec_beacon['Total'] = irec_beacon.swifter.apply(
        lambda x: x['Duration 0'] + x['Duration 1'] + x['Duration 2'] + x['Duration 3'] + x['Duration 4'] + x[
            'Duration 5'], axis=1)
    #irec_large = irec_beacon[irec_beacon['Total'] > 2]
    irec_beacon = irec_beacon[['Total']]
    irec_beacon.to_csv(os.path.join(irec_data_dir, 'handle_bcn.csv'), index=False)
    del irec_beacon

    # Propagation Filtering: ActionType.Propagated
    irec_propagation = irec_df[irec_df['Type'] == ActionType.Propagated.name]
    irec_propagation = irec_propagation[irec_propagation['Size'] == 3]
    # Data column as int
    irec_propagation['Data'] = irec_propagation['Data'].astype(int)
    irec_propagation = irec_propagation[['Duration 0', 'Duration 1', 'Duration 2', 'Data']]
    irec_propagation['Total'] = irec_propagation.swifter.apply(
        lambda x: (x['Duration 0'] + x['Duration 1'] + x['Duration 2']) / x['Data'], axis=1)
    irec_propagation = irec_propagation[['Total']]
    irec_propagation.to_csv(os.path.join(irec_data_dir, 'prop_filter.csv'), index=False)
    del irec_propagation

    # Propagation: ActionType.PropagatedBcn
    irec_propagation_bcn = irec_df[irec_df['Type'] == ActionType.PropagatedBcn.name]
    irec_propagation_bcn = irec_propagation_bcn[irec_propagation_bcn['Size'] == 5]
    irec_propagation_bcn = irec_propagation_bcn[['Duration 0', 'Duration 1', 'Duration 2', 'Duration 3', 'Duration 4']]
    # Maybe remove Duration 2 -> included in handle beacon
    irec_propagation_bcn['Total'] = irec_propagation_bcn.swifter.apply(
        #lambda x: x['Duration 0'] + x['Duration 1'] + x['Duration 2'] + x['Duration 3'] + x['Duration 4'], axis=1)
        lambda x: x['Duration 0'] + x['Duration 1'] + x['Duration 3'] + x['Duration 4'], axis = 1)
    irec_propagation_bcn = irec_propagation_bcn[['Total']]
    irec_propagation_bcn.to_csv(os.path.join(irec_data_dir, 'prop_bcn.csv'), index=False)
    del irec_propagation_bcn

    # Originate beacon: ActionType.OriginatedBcn
    irec_originated_bcn = irec_df[irec_df['Type'] == ActionType.OriginatedBcn.name]
    irec_originated_bcn = irec_originated_bcn[irec_originated_bcn['Size'] == 4]
    irec_originated_bcn = irec_originated_bcn[['Duration 0', 'Duration 1', 'Duration 2', 'Duration 3']]
    # Maybe remove Duration 2 -> included in handle beacon
    irec_originated_bcn['Total'] = irec_originated_bcn.swifter.apply(
        #lambda x: x['Duration 0'] + x['Duration 1'] + x['Duration 2'] + x['Duration 3'], axis=1)
        lambda x: x['Duration 0'] + x['Duration 1'] + x['Duration 3'], axis = 1)
    irec_originated_bcn = irec_originated_bcn[['Total']]
    irec_originated_bcn.to_csv(os.path.join(irec_data_dir, 'originated_bcn.csv'), index=False)
    del irec_originated_bcn

    # Job retrieval: ActionType.Retrieved
    irec_retrieval = irec_df[irec_df['Type'] == ActionType.Retrieved.name]
    irec_retrieval = irec_retrieval[irec_retrieval['Size'] == 3]
    # Data column as int
    irec_retrieval['Data'] = irec_retrieval['Data'].astype(int)
    irec_retrieval = irec_retrieval[['Duration 0', 'Duration 1', 'Duration 2', 'Data']]
    irec_retrieval['Total'] = irec_retrieval.swifter.apply(
        lambda x: (x['Duration 0'] + x['Duration 1'] + x['Duration 2']) / x['Data'], axis=1)
    irec_retrieval = irec_retrieval[['Total']]
    irec_retrieval.to_csv(os.path.join(irec_data_dir, 'job_retrieval.csv'), index=False)
    del irec_retrieval

    # Written: ActionType.Written
    irec_written = irec_df[irec_df['Type'] == ActionType.Written.name]
    irec_written = irec_written[irec_written['Size'] == 1]
    # Data column as int
    irec_written['Data'] = irec_written['Data'].astype(int)
    irec_written = irec_written[['Duration 0', 'Data']]
    irec_written['Total'] = irec_written.swifter.apply(lambda x: x['Duration 0'] / x['Data'], axis=1)
    irec_written = irec_written[['Total']]
    irec_written.to_csv(os.path.join(irec_data_dir, 'written.csv'), index=False)
    del irec_written

    # Mark egress: ActionType.Completed
    irec_egress = irec_df[irec_df['Type'] == ActionType.Completed.name]
    irec_egress = irec_egress[irec_egress['Size'] == 1]
    # Data column as int
    irec_egress['Data'] = irec_egress['Data'].astype(int)
    irec_egress = irec_egress[['Duration 0', 'Data']]
    irec_egress['Total'] = irec_egress.swifter.apply(lambda x: x['Duration 0'] / x['Data'], axis=1)
    irec_egress = irec_egress[['Total']]
    irec_egress.to_csv(os.path.join(irec_data_dir, 'egress_mark.csv'), index=False)
    del irec_egress

    # Algorithm retrieval: ActionType.Algorithm
    irec_algorithm = irec_df[irec_df['Type'] == ActionType.Algorithm.name]
    irec_algorithm = irec_algorithm[irec_algorithm['Size'] == 7]
    irec_algorithm = irec_algorithm[['Duration 0', 'Duration 1', 'Duration 2', 'Duration 3', 'Duration 4', 'Duration 5', 'Duration 6']]
    irec_algorithm['Total'] = irec_algorithm.swifter.apply(
        lambda x: x['Duration 0'] + x['Duration 1'] + x['Duration 2'] + x['Duration 3'] + x['Duration 4'] + x['Duration 5'] + x['Duration 6'], axis=1)
    irec_algorithm = irec_algorithm[['Total']]
    irec_algorithm.to_csv(os.path.join(irec_data_dir, 'algorithm.csv'), index=False)
    del irec_algorithm

    # Job processing: ActionType.Processed
    irec_processing = irec_df[irec_df['Type'] == ActionType.Processed.name]
    irec_processing = irec_processing[irec_processing['Size'] == 5]
    irec_processing = irec_processing[['Duration 0', 'Duration 1', 'Duration 2', 'Duration 3', 'Duration 4']]
    irec_processing['Total'] = irec_processing.swifter.apply(
        lambda x: x['Duration 0'] + x['Duration 1'] + x['Duration 2'] + x['Duration 3'] + x['Duration 4'], axis=1)
    irec_processing = irec_processing[['Total']]
    irec_processing.to_csv(os.path.join(irec_data_dir, 'job_processing.csv'), index=False)
    del irec_processing

    # Job execution: ActionType.Executed
    irec_execution = irec_df[irec_df['Type'] == ActionType.Executed.name]
    irec_execution = irec_execution[irec_execution['Size'] == 6]
    # Data column as int
    irec_execution['Data'] = irec_execution['Data'].astype(int)
    irec_execution = irec_execution[['Duration 0', 'Duration 1', 'Duration 2', 'Duration 3', 'Duration 4', 'Duration 5', 'Data']]
    irec_execution['Total'] = irec_execution.swifter.apply(
        lambda x: (x['Duration 0'] + x['Duration 1'] + x['Duration 2'] + x['Duration 3'] + x['Duration 4'] + x['Duration 5']) / x['Data'], axis=1)
    irec_execution = irec_execution[['Total']]
    irec_execution.to_csv(os.path.join(irec_data_dir, 'job_execution.csv'), index=False)
    del irec_execution

    # DONE
    del irec_df

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

    # for scion_name in scion_dir_names:
    #     scion_to_data(scion_name)

    for irec_name in irec_dir_names:
        irec_to_data(irec_name)

    pass
