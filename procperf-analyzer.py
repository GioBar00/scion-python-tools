import os
from datetime import timedelta
from pathlib import Path
import enum
import pandas as pd
import matplotlib.pyplot as plt
from fontTools.misc.cython import returns
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
    Originated = 1
    Propagated = 2
    Processed = 3


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


# def originated_beacons(df: pd.DataFrame, cs: str) -> pd.DataFrame:
#     return df[(df['Type'] == ActionType.Originated.name) & (df['Device'] == cs)]
#
#
# def propagated_original_device(df: pd.DataFrame, unreached_cs: set[str]) -> pd.DataFrame:
#     if unreached_cs is None or len(unreached_cs) == 0:
#         return df
#     df_join = df[(df['Type'] != ActionType.Originated.name) & (df['Original Device'] == "")].merge(df[(df['Type'] != ActionType.Received.name) & (df['Original Device'] != "")], left_on='ID', right_on='Next ID', suffixes=('_next', '_prev'))
#     if df_join.empty:
#         return df
#     df_join['Original Device_next'] = df_join['Original Device_prev']
#     # Keep only _next columns
#     df_join = df_join[[col for col in df_join.columns if col.endswith("_next")]]
#     # Remove _next suffix
#     df_join.columns = [col[:-5] for col in df_join.columns]
#     # Left join to original dataframe on all columns except Original Device
#     df = df.merge(df_join, on=[col for col in df.columns if col != "Original Device"], how='left', suffixes=('', '_new'))
#     # Fill Original Device with Original Device_next
#     df['Original Device'] = df['Original Device'].mask(df['Original Device'] == "", df['Original Device_new'])
#     # Drop Original Device_new
#     df = df.drop(columns=['Original Device_new'])
#     # Convert nan to empty string
#     df['Original Device'] = df['Original Device'].fillna("")
#     return propagated_original_device(df, set(unreached_cs) - set(df_join['Device']))
#
#
# def complete_as_propagation(df: pd.DataFrame, a_s: str, shared_seg_ids: set[str]=None) -> pd.Timedelta:
#     cs = a_s.lower().replace(":", "_").replace("-", "_")
#     if not cs.startswith("cs"):
#         cs = f"cs{cs}_1"
#     orig_beacons = originated_beacons(df, cs)
#     min_time = orig_beacons['Start Time'].min()
#
#     if shared_seg_ids is None:
#         shared_seg_ids = set()
#
#     seg_ids = orig_beacons['SegID'].unique()
#     seg_ids = set(seg_ids) - shared_seg_ids
#
#     df_filtered = df[(df['Device'] != cs) & (df['Type'] == ActionType.Received.name) & (df['SegID'].isin(seg_ids))]
#     cses = df_filtered['Device'].unique()
#     prop_times = {cs: df_filtered[df_filtered['Device'] == cs]['End Time'].min() for cs in cses}
#
#     # Check for shared segIDs
#     df_filtered = df[df['SegID'].isin(shared_seg_ids)]
#     # If type is Originated, keep only rows with Device == cs
#     df_filtered = df_filtered[(df_filtered['Type'] != ActionType.Originated.name) | (df_filtered['Device'] == cs)]
#     df_filtered['Original Device'] = pd.Series("", index=df_filtered.index).mask(df_filtered['Type'] == ActionType.Originated.name, df_filtered['Device'])
#     df_filtered = propagated_original_device(df_filtered, cses)
#     df_filtered = df_filtered[df_filtered['Original Device'] != ""]
#     for cs in df_filtered['Device'].unique():
#         if cs not in prop_times:
#             prop_times[cs] = df_filtered[df_filtered['Device'] == cs]['End Time'].min()
#         else:
#             prop_times[cs] = min(prop_times[cs], df_filtered[df_filtered['Device'] == cs]['End Time'].min())
#
#     max_time = max(prop_times.values())
#     return max_time - min_time
#
#
# def complete_propagation(df: pd.DataFrame) -> timedelta:
#     df_orig = df[df['Type'] == ActionType.Originated.name]
#     # get SegIDs that are used by different devices
#     shared_seg_ids = df_orig.groupby('SegID').filter(lambda x: len(x['Device'].unique()) > 1)['SegID'].unique()
#     shared_seg_ids = set(shared_seg_ids)
#     elapsed = [complete_as_propagation(df, cs, shared_seg_ids).to_pytimedelta() for cs in scion_cses]
#     return max(elapsed)


def plot_beacon_handling(df_scion, df_irec, df_irec_rac):
    if df_scion is None:
        # Load from csv
        df_scion_grouped = pd.read_csv("scion_grouped.csv")
        df_irec_grouped = pd.read_csv("irec_grouped.csv")
        df_irec_rac_grouped = pd.read_csv("irec_rac_grouped.csv")
    else:
        # Keep only Type and Duration columns
        df_scion_grouped = df_scion[['Type', 'Duration']].groupby('Type').mean()
        df_irec_grouped = df_irec[['Type', 'Duration']].groupby('Type').mean()
        df_irec_rac_grouped = df_irec_rac[['Type', 'Duration']].groupby('Type').mean()
        # Save to csv
        # df_scion_grouped.to_csv("scion_grouped.csv")
        # df_irec_grouped.to_csv("irec_grouped.csv")
        # df_irec_rac_grouped.to_csv("irec_rac_grouped.csv")

    # Convert Duration to Timedelta
    df_scion_grouped['Duration'] = pd.to_timedelta(df_scion_grouped['Duration'])
    df_irec_grouped['Duration'] = pd.to_timedelta(df_irec_grouped['Duration'])
    df_irec_rac_grouped['Duration'] = pd.to_timedelta(df_irec_rac_grouped['Duration'])

    # Make index the Type column
    df_scion_grouped = df_scion_grouped.set_index('Type')
    df_irec_grouped = df_irec_grouped.set_index('Type')
    df_irec_rac_grouped = df_irec_rac_grouped.set_index('Type')

    df_irec_grouped = pd.concat([df_irec_grouped, df_irec_rac_grouped])

    # Order by ActionType enum
    df_scion_grouped = df_scion_grouped.reindex([t.name for t in ActionType if t.name in df_scion_grouped.index])
    df_irec_grouped = df_irec_grouped.reindex([t.name for t in ActionType if t.name in df_irec_grouped.index])

    fig, ax = plt.subplots()
    width = 0.35
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
    scion_dev_cpu = {d: np.array([]) for d in DeviceType if d != DeviceType.RAC}
    scion_dev_mem = {d: np.array([]) for d in DeviceType if d != DeviceType.RAC}
    irec_dev_cpu = {d: np.array([]) for d in DeviceType}
    irec_dev_mem = {d: np.array([]) for d in DeviceType}

    def df_to_dev_dict(df: pd.DataFrame, dev_dict: dict):
        for dev in df.columns:
            dev_type = to_device_type(dev)
            values = df[dev].dropna().values
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



if __name__ == '__main__':
    """
    scion_dir = os.path.join(os.getcwd(), f"procperf-{TOPO}/")
    irec_dir = os.path.join(os.getcwd(), f"procperf-{TOPO}-irec-{CP.name.lower()}/")

    scion_cses = [Path(f).stem for f in os.listdir(scion_dir) if f.endswith(".csv")]
    irec_cses = [Path(f).stem for f in os.listdir(irec_dir) if f.endswith(".csv") and f.startswith("cs")]
    irec_racs = [Path(f).stem for f in os.listdir(irec_dir) if f.endswith(".csv") and f.startswith("rac")]

    # Check scion_cses and irec_cses are the same values
    assert scion_cses == irec_cses

    dfs_scion = [pd.read_csv(os.path.join(scion_dir, f"{dev}.csv"), sep=";") for dev in scion_cses]
    for i in range(len(dfs_scion)):
        # remove leading and trailing whitespaces from column names
        #dfs_scion[i].columns = dfs_scion[i].columns.str.strip()
        # remove leading and trailing whitespaces from values
        #dfs_scion[i] = dfs_scion[i].map(lambda x: x.strip() if isinstance(x, str) else x)
        dfs_scion[i]["Device"] = scion_cses[i]

    df_scion = pd.concat(dfs_scion)
    df_scion['SegID'] = df_scion['ID'].map(lambda x: x.split(" ")[1])

    # iterate over irec_cses and irec_racs
    dfs_irec = [pd.read_csv(os.path.join(irec_dir, f"{dev}.csv"), sep=";") for dev in irec_cses]
    for i in range(len(dfs_irec)):
        # remove leading and trailing whitespaces from column names
        #dfs_irec[i].columns = dfs_irec[i].columns.str.strip()
        # remove leading and trailing whitespaces from values
        #dfs_irec[i] = dfs_irec[i].map(lambda x: x.strip() if isinstance(x, str) else x)
        dfs_irec[i]["Device"] = irec_cses[i]

    df_irec = pd.concat(dfs_irec)
    df_irec['SegID'] = df_irec['ID'].map(lambda x: x.split(" ")[1])

    dfs_irec_rac = [pd.read_csv(os.path.join(irec_dir, f"{dev}.csv"), sep=";") for dev in irec_racs]
    for i in range(len(dfs_irec_rac)):
        # remove leading and trailing whitespaces from column names
        #dfs_irec_rac[i].columns = dfs_irec_rac[i].columns.str.strip()
        # remove leading and trailing whitespaces from values
        #dfs_irec_rac[i] = dfs_irec_rac[i].map(lambda x: x.strip() if isinstance(x, str) else x)
        dfs_irec_rac[i]["Device"] = irec_racs[i]

    df_irec_rac = pd.concat(dfs_irec_rac)

    # convert to datetime
    df_scion['Start Time'] = pd.to_datetime(df_scion['Start Time'])
    df_irec['Start Time'] = pd.to_datetime(df_irec['Start Time'])
    df_irec_rac['Start Time'] = pd.to_datetime(df_irec_rac['Start Time'])
    df_scion['End Time'] = pd.to_datetime(df_scion['End Time'])
    df_irec['End Time'] = pd.to_datetime(df_irec['End Time'])
    df_irec_rac['End Time'] = pd.to_datetime(df_irec_rac['End Time'])

    # Add duration column
    df_scion['Duration'] = df_scion['End Time'] - df_scion['Start Time']
    df_irec['Duration'] = df_irec['End Time'] - df_irec['Start Time']
    df_irec_rac['Duration'] = df_irec_rac['End Time'] - df_irec_rac['Start Time']

    # elapsed = complete_propagation(df_scion)
    # print("Max elapsed time for complete connectivity:", elapsed)
    # elapsed = complete_propagation(df_irec)
    # print("Max elapsed time for complete connectivity (IREC):", elapsed)
    
    plot_beacon_handling(df_scion, df_irec, df_irec_rac)
    """
    plot_beacon_handling(None, None, None)

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






