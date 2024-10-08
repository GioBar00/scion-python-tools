import datetime
import os
from datetime import timedelta, datetime
from pathlib import Path
import enum
import pandas as pd
# import ray
# ray.init(num_cpus=6)
# import modin.pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import humanfriendly
import numpy as np


class ControlPlaneType(enum.Enum):
    NATIVE = 0
    UBPF = 1
    UBPFJIT = 2
    WA = 3
    WAOPT = 4


class ActionType(enum.Enum):
    Received = 0
    Propagated = 1
    Originated = 2
    Retrieved = 3
    Written = 4
    Processed = 5
    Executed = 6
    Completed = 7
    Algorithm = 8


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


class DeviceType(enum.Enum):
    CS = "cs"
    BR = "br"
    SD = "sd"
    RAC = "rac"


TOPO = "tier1"
CP = ControlPlaneType.UBPF


def originated_beacons(df: pd.DataFrame, cs: str) -> pd.DataFrame:
    return df[(df['Type'] == ActionType.Originated.name) & (df['Device'] == cs)]


def propagated_original_device(df: pd.DataFrame, unreached_cs: set[str]) -> pd.DataFrame:
    if unreached_cs is None or len(unreached_cs) == 0:
        return df
    df_join = df[(df['Type'] != ActionType.Originated.name) & (df['Original Device'] == "")].merge(df[(df['Type'] != ActionType.Received.name) & (df['Original Device'] != "")], left_on='ID', right_on='Next ID', suffixes=('_next', '_prev'))
    if df_join.empty:
        return df
    df_join['Original Device_next'] = df_join['Original Device_prev']
    # Keep only _next columns
    df_join = df_join[[col for col in df_join.columns if col.endswith("_next")]]
    # Remove _next suffix
    df_join.columns = [col[:-5] for col in df_join.columns]
    # Left join to original dataframe on all columns except Original Device
    df = df.merge(df_join, on=[col for col in df.columns if col != "Original Device"], how='left', suffixes=('', '_new'))
    # Fill Original Device with Original Device_next
    df['Original Device'] = df['Original Device'].mask(df['Original Device'] == "", df['Original Device_new'])
    # Drop Original Device_new
    df = df.drop(columns=['Original Device_new'])
    # Convert nan to empty string
    df['Original Device'] = df['Original Device'].fillna("")
    return propagated_original_device(df, set(unreached_cs) - set(df_join['Device']))


def complete_as_propagation(df: pd.DataFrame, a_s: str, shared_seg_ids: set[str]=None) -> pd.Timedelta:
    cs = a_s.lower().replace(":", "_").replace("-", "_")
    if not cs.startswith("cs"):
        cs = f"cs{cs}_1"
    print(f"Processing {cs}")
    orig_beacons = originated_beacons(df, cs)
    min_time = orig_beacons['Start Time'].min()

    if shared_seg_ids is None:
        shared_seg_ids = set()

    seg_ids = orig_beacons['SegID'].unique()
    seg_ids = set(seg_ids) - shared_seg_ids

    df_filtered = df[(df['Device'] != cs) & (df['Type'] == ActionType.Received.name) & (df['SegID'].isin(seg_ids))]
    cses = df_filtered['Device'].unique()
    prop_times = {cs: df_filtered[df_filtered['Device'] == cs]['End Time'].min() for cs in cses}

    print("Checking for shared segIDs")
    # Check for shared segIDs
    df_filtered = df[df['SegID'].isin(shared_seg_ids)]
    # If type is Originated, keep only rows with Device == cs
    df_filtered = df_filtered[(df_filtered['Type'] != ActionType.Originated.name) | (df_filtered['Device'] == cs)]
    df_filtered['Original Device'] = pd.Series("", index=df_filtered.index).mask(df_filtered['Type'] == ActionType.Originated.name, df_filtered['Device'])
    df_filtered = propagated_original_device(df_filtered, cses)
    df_filtered = df_filtered[df_filtered['Original Device'] != ""]
    for cs in df_filtered['Device'].unique():
        if cs not in prop_times:
            prop_times[cs] = df_filtered[df_filtered['Device'] == cs]['End Time'].min()
        else:
            prop_times[cs] = min(prop_times[cs], df_filtered[df_filtered['Device'] == cs]['End Time'].min())

    max_time = max(prop_times.values())
    return max_time - min_time


def complete_propagation(df: pd.DataFrame) -> timedelta:
    # Keep only Originated, Received and Propagated types
    df = df[df['Type'].isin([ActionType.Originated.name, ActionType.Received.name, ActionType.Propagated.name])]
    df = df.copy()
    # Add Start Time column
    df['Start Time'] = df['Time Array'].map(lambda x: x[0])
    # Add End Time column
    df['End Time'] = df['Time Array'].map(lambda x: x[-1])
    # Remove Time Array column
    df = df.drop(columns=['Time Array'])
    # Add SegID column
    df['SegID'] = df['ID'].map(lambda x: x.split(" ")[1])
    df_orig = df[df['Type'] == ActionType.Originated.name]
    # get SegIDs that are used by different devices
    shared_seg_ids = df_orig.groupby('SegID').filter(lambda x: len(x['Device'].unique()) > 1)['SegID'].unique()
    shared_seg_ids = set(shared_seg_ids)
    print("Shared SegIDs len: ", len(shared_seg_ids))
    elapsed = [complete_as_propagation(df, cs, shared_seg_ids).to_pytimedelta() for cs in scion_cses]
    return max(elapsed)



def get_data_irec():
    result = {}
    # Compute Received
    print("IREC Received")
    series = df_irec[df_irec['Type'] == ActionType.Received.name]['Time Array']
    result['Received'] = {}
    result['Received']['Duration'] = series.map(lambda x: x[-1]) - series.map(lambda x: x[0])
    result['Received']['Details'] = {}
    result['Received']['Details']['Validate'] = series.map(lambda x: x[2]) - series.map(lambda x: x[1])
    result['Received']['Details']['Verify'] = series.map(lambda x: x[3]) - series.map(lambda x: x[2])
    result['Received']['Details']['PreFilter'] = series.map(lambda x: x[4]) - series.map(lambda x: x[3])
    result['Received']['Details']['ValAlg'] = series.map(lambda x: x[5]) - series.map(lambda x: x[4])
    result['Received']['Details']['Insert'] = series.map(lambda x: x[7]) - series.map(lambda x: x[6])
    # Compute Propagated
    print("IREC Propagated")
    series = df_irec[df_irec['Type'] == ActionType.Propagated.name]['Time Array']
    result['Propagated'] = {}
    result['Propagated']['Duration'] = series.map(lambda x: x[-1]) - series.map(lambda x: x[0])
    result['Propagated']['Details'] = {}
    result['Propagated']['Details']['Propagate'] = series.map(lambda x: x[-1]) - series.map(lambda x: x[1])
    result['Propagated']['Details']['Check'] = series.map(lambda x: x[3]) - series.map(lambda x: x[2])
    result['Propagated']['Details']['Filter'] = series.map(lambda x: x[4]) - series.map(lambda x: x[3])
    result['Propagated']['Details']['Extend'] = series.map(lambda x: x[5]) - series.map(lambda x: x[4])
    result['Propagated']['Details']['PreMark'] = series.map(lambda x: x[6]) - series.map(lambda x: x[5])
    result['Propagated']['Details']['Sender'] = series.map(lambda x: x[7]) - series.map(lambda x: x[6])
    result['Propagated']['Details']['Send'] = series.map(lambda x: x[8]) - series.map(lambda x: x[7])
    result['Propagated']['Details']['Mark'] = series.map(lambda x: x[9]) - series.map(lambda x: x[8])
    # Compute Originated
    print("IREC Originated")
    series = df_irec[df_irec['Type'] == ActionType.Originated.name]['Time Array']
    result['Originated'] = {}
    result['Originated']['Duration'] = series.map(lambda x: x[-1]) - series.map(lambda x: x[0])
    result['Originated']['Details'] = {}
    result['Originated']['Details']['Sender'] = series.map(lambda x: x[1]) - series.map(lambda x: x[0])
    result['Originated']['Details']['Create'] = series.map(lambda x: x[3]) - series.map(lambda x: x[2])
    result['Originated']['Details']['Send'] = series.map(lambda x: x[5]) - series.map(lambda x: x[4])
    # Compute Retrieved
    print("IREC Retrieved")
    series = df_irec[df_irec['Type'] == ActionType.Retrieved.name]['Time Array']
    result['Retrieved'] = {}
    result['Retrieved']['Duration'] = series.map(lambda x: x[-1]) - series.map(lambda x: x[0])
    result['Retrieved']['Details'] = {}
    result['Retrieved']['Details']['DbGetJob'] = series.map(lambda x: x[1]) - series.map(lambda x: x[0])
    # Compute Written
    print("IREC Written")
    df = df_irec[df_irec['Type'] == ActionType.Written.name]
    result['Written'] = {}
    result['Written']['Duration'] = None
    result['Written']['Details'] = {}
    # Group by ID
    ids = df['ID'].unique()
    for id in ids:
        series = df[df['ID'] == id]['Time Array']
        result['Written']['Details'][id] = series.map(lambda x: x[-1]) - series.map(lambda x: x[0])
    # Compute Completed
    print("IREC Completed")
    series = df_irec[df_irec['Type'] == ActionType.Completed.name]['Time Array']
    result['Completed'] = {}
    result['Completed']['Duration'] = series.map(lambda x: x[-1]) - series.map(lambda x: x[0])
    result['Completed']['Details'] = {}
    # Compute Algorithm
    print("IREC Algorithm")
    series = df_irec[df_irec['Type'] == ActionType.Algorithm.name]['Time Array']
    result['Algorithm'] = {}
    result['Algorithm']['Duration'] = None
    result['Algorithm']['Details'] = {}
    series_alg_exists = series[series.map(lambda x: len(x) == 2)]
    result['Algorithm']['Details']['Exists'] = series_alg_exists.map(lambda x: x[1]) - series_alg_exists.map(lambda x: x[0])
    series_alg_not_exists = series[series.map(lambda x: len(x) > 2)]
    result['Algorithm']['Details']['NotExists'] = series_alg_not_exists.map(lambda x: x[-1]) - series_alg_not_exists.map(lambda x: x[0])
    # TODO: Show not exists details

    return result

def get_data_irec_rac():
    result = {}
    # Compute Processed
    print("IREC RAC Processed")
    series = df_irec_rac[df_irec_rac['Type'] == ActionType.Processed.name]['Time Array']
    result['Processed'] = {}
    result['Processed']['Duration'] = series.map(lambda x: x[-1]) - series.map(lambda x: x[0])
    result['Processed']['Details'] = {}
    # Infer type from time array length
    rac_type = "Dynamic" if series.map(lambda x: len(x) == 6).all() else "Static"
    result['Processed']['Details']['Type'] = rac_type
    result['Processed']['Details']['GetJob'] = series.map(lambda x: x[1]) - series.map(lambda x: x[0])
    if rac_type == "Dynamic":
        result['Processed']['Details']['GetAlg'] = series.map(lambda x: x[3]) - series.map(lambda x: x[2])
        result['Processed']['Details']['ExecAlg'] = series.map(lambda x: x[4]) - series.map(lambda x: x[3])
    else:
        result['Processed']['Details']['ExecAlg'] = series.map(lambda x: x[3]) - series.map(lambda x: x[2])
    result['Processed']['Details']['JobComplete'] = series.map(lambda x: x[-1]) - series.map(lambda x: x[-2])
    # Compute Executed
    print("IREC RAC Executed")
    series = df_irec_rac[df_irec_rac['Type'] == ActionType.Executed.name]['Time Array']
    result['Executed'] = {}
    result['Executed']['Duration'] = series.map(lambda x: x[-1]) - series.map(lambda x: x[0])
    result['Executed']['Details'] = {}
    result['Executed']['Details']['Create'] = series.map(lambda x: x[1]) - series.map(lambda x: x[0])
    result['Executed']['Details']['PrepMem'] = series.map(lambda x: x[2]) - series.map(lambda x: x[1])
    result['Executed']['Details']['Exec'] = series.map(lambda x: x[3]) - series.map(lambda x: x[2])
    result['Executed']['Details']['LoadRes'] = series.map(lambda x: x[4]) - series.map(lambda x: x[3])
    # result['Executed']['Details']['DestrMem'] = series.map(lambda x: x[5]) - series.map(lambda x: x[4])
    # result['Executed']['Details']['ReqProp'] = series.map(lambda x: x[6]) - series.map(lambda x: x[5])
    #TODO: change with above
    result['Executed']['Details']['ReqProp'] = series.map(lambda x: x[5]) - series.map(lambda x: x[4])
    return result


def plot_beacon_handling():
    result_irec = get_data_irec()

    fig, ax = plt.subplots()
    width = 0.35
    valid_keys = [key for key in result_irec.keys() if result_irec[key]['Duration'] is not None and key != ActionType.Propagated.name]
    x = range(len(valid_keys))
    x_irec = [i for i in x]
    ax.boxplot([result_irec[key]['Duration'].dt.total_seconds() for key in valid_keys], positions=x_irec, widths=width, showfliers=False, showmeans=True)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_keys)
    ax.set_ylabel('Duration (s)')
    ax.set_title('Average duration of beacon handling')
    plt.show()

    # Show Propagated details
    fig, ax = plt.subplots()
    width = 0.35
    valid_keys = [key for key in result_irec[ActionType.Propagated.name]['Details'].keys()]
    x = range(len(valid_keys))
    x_irec = [i for i in x]
    ax.boxplot([result_irec[ActionType.Propagated.name]['Details'][key].dt.total_seconds() for key in valid_keys], positions=x_irec, widths=width, showfliers=False, showmeans=True)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_keys)
    ax.set_ylabel('Duration (s)')
    ax.set_title(ActionType.Propagated.name)
    plt.show()
    del result_irec

    result_irec_rac = get_data_irec_rac()
    fig, ax = plt.subplots()
    width = 0.35
    valid_keys = [key for key in result_irec_rac.keys() if result_irec_rac[key]['Duration'] is not None]
    x = range(len(valid_keys))
    x_irec = [i for i in x]
    ax.boxplot([result_irec_rac[key]['Duration'].dt.total_seconds() for key in valid_keys], positions=x_irec, widths=width, showfliers=False, showmeans=True)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_keys)
    ax.set_ylabel('Duration (s)')
    ax.set_title('RAC Average duration of beacon handling')
    plt.show()

    # Show Processed details
    fig, ax = plt.subplots()
    width = 0.35
    valid_keys = [key for key in result_irec_rac[ActionType.Processed.name]['Details'].keys() if key != 'Type']
    x = range(len(valid_keys))
    x_irec = [i for i in x]
    ax.boxplot([result_irec_rac[ActionType.Processed.name]['Details'][key].dt.total_seconds() for key in valid_keys], positions=x_irec, widths=width, showfliers=False, showmeans=True)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_keys)
    ax.set_ylabel('Duration (s)')
    ax.set_title(ActionType.Processed.name + " - " + result_irec_rac[ActionType.Processed.name]['Details']['Type'])
    plt.show()

    # Show Executed details
    fig, ax = plt.subplots()
    width = 0.35
    valid_keys = [key for key in result_irec_rac[ActionType.Executed.name]['Details'].keys()]
    x = range(len(valid_keys))
    x_irec = [i for i in x]
    ax.boxplot([result_irec_rac[ActionType.Executed.name]['Details'][key].dt.total_seconds() for key in valid_keys], positions=x_irec, widths=width, showfliers=False, showmeans=True)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_keys)
    ax.set_ylabel('Duration (s)')
    ax.set_title(ActionType.Executed.name)
    plt.show()

    del result_irec_rac

    return
    # Originated, Received and Propagated types have 2 bars (scion, irec), Processed has 1 (irec_rac)
    x = range(len(df_scion_grouped.index) + 1)
    x_scion = [i - width / 2 for i in x[:-1]]
    x_irec = [i + width / 2 for i in x[:-1]] + [x[-1]]

    ax.bar(x_scion, df_scion_grouped['Duration'].dt.total_seconds(), width, label='SCION')
    ax.bar(x_irec, df_irec_grouped['Duration'].dt.total_seconds(), width, label='IREC')

    ax.set_xticks(x)
    ax.set_xticklabels([t.name for t in ActionType])
    ax.set_ylabel('Duration (s)')
    ax.set_title('Average duration of beacon handling')
    ax.legend()

    # Add numbers on top of bars
    for i, v in enumerate(df_scion_grouped['Duration']):
        ax.text(x_scion[i], v.total_seconds(), "%.3f" % v.total_seconds(), ha='center', va='bottom')
    for i, v in enumerate(df_irec_grouped['Duration']):
        ax.text(x_irec[i], v.total_seconds(), "%.3f" % v.total_seconds(), ha='center', va='bottom')

    plt.show()


def plot_cpu_memory_usage_cdf(df_cpu_scion, df_mem_scion, df_cpu_irec, df_mem_irec):
    scion_dev_cpu = {d: np.array([]) for d in DeviceType if d != DeviceType.RAC and d != DeviceType.SD}
    scion_dev_mem = {d: np.array([]) for d in DeviceType if d != DeviceType.RAC and d != DeviceType.SD}
    irec_dev_cpu = {d: np.array([]) for d in DeviceType if d != DeviceType.SD}
    irec_dev_mem = {d: np.array([]) for d in DeviceType if d != DeviceType.SD}

    def df_to_dev_dict(df: pd.DataFrame, dev_dict: dict):
        for dev in df.columns:
            dev_type = to_device_type(dev)
            values = df[dev].dropna().values
            if dev_type in dev_dict:
                dev_dict[dev_type] = np.append(dev_dict[dev_type], values)

    df_to_dev_dict(df_cpu_scion, scion_dev_cpu)
    df_to_dev_dict(df_mem_scion, scion_dev_mem)
    df_to_dev_dict(df_cpu_irec, irec_dev_cpu)
    df_to_dev_dict(df_mem_irec, irec_dev_mem)

    def plot_usage(title: str, x_label: str, scion_dev_dict: dict, irec_dev_dict: dict):
        # Plot CPU usage CDFs for each device type comparing SCION and IREC
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(title)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, dev_type in enumerate(DeviceType):
            if dev_type == DeviceType.SD:
                continue
            ax = axs[i % 2, i // 2]
            ax.set_title(dev_type.name)
            unit = 1
            new_x_label = x_label
            if title.startswith("Memory"):
                # Change ticks to human-readable format
                if dev_type != DeviceType.RAC:
                    min_value = min(np.mean(scion_dev_dict[dev_type]), np.mean(irec_dev_dict[dev_type]))
                else:
                    min_value = np.mean(irec_dev_dict[dev_type])
                min_value = int(min_value)
                unit = humanfriendly.format_size(min_value, binary=True).split(" ")[1]
                new_x_label = f"{x_label} [{unit}]"
                unit = humanfriendly.parse_size(f"1 {unit}")

            ax.set_xlabel(new_x_label)
            ax.set_ylabel('$p$')
            if dev_type != DeviceType.RAC:
                sorted_scion = np.sort(scion_dev_dict[dev_type]) / unit
                p = 1. * np.arange(len(sorted_scion)) / (len(sorted_scion) - 1)
                ax.plot(sorted_scion, p, label='SCION', color=colors[0])
                # Print 95th percentile
                percentile = np.percentile(scion_dev_dict[dev_type], 95) / unit
                ax.axvline(percentile, color=colors[0], linestyle='--')
                offset = 0.02 * sorted_scion[-1]
                h_align = 'left'
                if percentile / sorted_scion[-1] > 0.85:
                    offset *= -1
                    h_align = 'right'
                ax.text(percentile + offset, 0.1, f'95th:\n{percentile:.2f}', color=colors[0],
                        horizontalalignment=h_align)

            sorted_irec = np.sort(irec_dev_dict[dev_type]) / unit
            p = 1. * np.arange(len(sorted_irec)) / (len(sorted_irec) - 1)
            ax.plot(sorted_irec, p, label='IREC', color=colors[1])
            # Print 95th percentile
            percentile = np.percentile(irec_dev_dict[dev_type], 95) / unit
            ax.axvline(percentile, color=colors[1], linestyle='--')
            offset = 0.02 * sorted_irec[-1]
            h_align = 'left'
            if percentile / sorted_irec[-1] > 0.85:
                offset *= -1
                h_align = 'right'
            ax.text(percentile + offset, 0, f'95th:\n{percentile:.2f}', color=colors[1], horizontalalignment=h_align)
            ax.legend()

        plt.show()

    plot_usage("CPU Usage CDF", "CPU Usage [cores]", scion_dev_cpu, irec_dev_cpu)
    plot_usage("Memory Usage CDF", "Memory Usage", scion_dev_mem, irec_dev_mem)


def load_df(devs_dir: str, devs: list[str]) -> pd.DataFrame:
    dtype = {col: str for col in ['Type', 'ID', 'Next ID', 'Time Array']}
    dfs = [pd.read_csv(os.path.join(devs_dir, f"{dev}.csv"), sep=";", dtype=dtype) for dev in devs]
    for i in range(len(dfs)):
        dfs[i]["Device"] = devs[i]
    res = pd.concat(dfs)
    del dfs
    # convert to datetime
    # def series_to_datetime(series: pd.Series) -> pd.Series:
    #     # series = series.str[1:-1].str.split(",")
    #     # series = series.map(lambda x: x[1:-1].split(","))
    #     # return series.map(lambda x: [datetime.fromisoformat(s.strip()) for s in x])
    #     return series.map(lambda x: [datetime.fromisoformat(s.strip()) for s in x[1:-1].split(",")])
    # res['Time Array'] = series_to_datetime(res['Time Array'])
    res['Time Array'] = res['Time Array'].map(lambda x: [datetime.fromisoformat(s.strip()) for s in x[1:-1].split(",") if s.strip() != ""])
    return res


if __name__ == '__main__':
    scion_dir = os.path.join(os.getcwd(), f"procperf-{TOPO}/")
    #irec_dir = os.path.join(os.getcwd(), f"procperf-{TOPO}-irec-{CP.name.lower()}/")
    irec_dir = os.path.join(os.getcwd(), f"procperf-{TOPO}-irec-{CP.name.lower()}-indexed-fix-nopool/")

    scion_cses = [Path(f).stem for f in os.listdir(scion_dir) if f.endswith(".csv")]
    irec_cses = [Path(f).stem for f in os.listdir(irec_dir) if f.endswith(".csv") and f.startswith("cs")]
    irec_racs = [Path(f).stem for f in os.listdir(irec_dir) if f.endswith(".csv") and f.startswith("rac")]

    # Check scion_cses and irec_cses are the same values
    assert scion_cses == irec_cses
    print("SCION and IREC CSEs are the same")

    # print("Loading SCION")
    # df_scion = load_df(scion_dir, scion_cses)
    print("Loading IREC")
    df_irec = load_df(irec_dir, irec_cses)
    print("Loading IREC RAC")
    df_irec_rac = load_df(irec_dir, irec_racs)

    # print("start complete propagation")
    # elapsed = complete_propagation(df_scion)
    # print("Max elapsed time for complete connectivity:", elapsed)
    # elapsed = complete_propagation(df_irec)
    # print("Max elapsed time for complete connectivity (IREC):", elapsed)

    # df_scion_prop = df_scion[df_scion['Type'] == ActionType.Propagated.name]
    # df_irec_prop = df_irec[df_irec['Type'] == ActionType.Propagated.name]
    #
    # print("SCION Propagated: ", len(df_scion_prop))
    # print("IREC Propagated: ", len(df_irec_prop))
    
    # plot_beacon_handling(df_scion, df_irec, df_irec_rac)
    # plot_beacon_handling(None, None, None)
    plot_beacon_handling()



    exit(0)
    usage_scion_dir = os.path.join(os.getcwd(), f"usage-{TOPO}/")
    usage_irec_dir = os.path.join(os.getcwd(), f"usage-{TOPO}-irec-{CP.name.lower()}/")

    df_cpu_scion = pd.read_csv(os.path.join(usage_scion_dir, f"cpu-usage-{TOPO}.csv"))
    df_mem_scion = pd.read_csv(os.path.join(usage_scion_dir, f"memory-usage-{TOPO}.csv"))
    df_cpu_irec = pd.read_csv(os.path.join(usage_irec_dir, f"cpu-usage-{TOPO}-irec-{CP.name.lower()}.csv"))
    df_mem_irec = pd.read_csv(os.path.join(usage_irec_dir, f"memory-usage-{TOPO}-irec-{CP.name.lower()}.csv"))

    # Parse Column 'Time' to datetime
    df_cpu_scion['Time'] = pd.to_datetime(df_cpu_scion['Time'])
    df_mem_scion['Time'] = pd.to_datetime(df_mem_scion['Time'])
    df_cpu_irec['Time'] = pd.to_datetime(df_cpu_irec['Time'])
    df_mem_irec['Time'] = pd.to_datetime(df_mem_irec['Time'])

    # Set 'Time' as index
    df_cpu_scion = df_cpu_scion.set_index('Time')
    df_mem_scion = df_mem_scion.set_index('Time')
    df_cpu_irec = df_cpu_irec.set_index('Time')
    df_mem_irec = df_mem_irec.set_index('Time')

    def pod_to_dev(pod: str) -> str:
        pod = pod.replace("kathara-", "")
        parts = str.split(pod, "-")
        if parts[0].startswith("sd"):
            return '_'.join(parts[:4])
        return '_'.join(parts[:5])


    # Rename columns except 'Time' using regex
    df_cpu_scion = df_cpu_scion.rename(columns=lambda x: pod_to_dev(x))
    df_mem_scion = df_mem_scion.rename(columns=lambda x: pod_to_dev(x))
    df_cpu_irec = df_cpu_irec.rename(columns=lambda x: pod_to_dev(x))
    df_mem_irec = df_mem_irec.rename(columns=lambda x: pod_to_dev(x))

    # Convert memory usage from human-readable format
    df_mem_scion = df_mem_scion.map(lambda x: humanfriendly.parse_size(x) if isinstance(x, str) else x)
    df_mem_irec = df_mem_irec.map(lambda x: humanfriendly.parse_size(x) if isinstance(x, str) else x)

    plot_cpu_memory_usage_cdf(df_cpu_scion, df_mem_scion, df_cpu_irec, df_mem_irec)






