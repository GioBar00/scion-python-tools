import enum
import os.path
from pathlib import Path
import pandas as pd
import shutil
import swifter
import numpy as np
import humanfriendly
from matplotlib import pyplot as plt

OVERRIDE = True

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


def init_plot_dir(name):
    sub_plot_dir = os.path.join(plot_dir, name)
    if not os.path.exists(sub_plot_dir):
        os.makedirs(sub_plot_dir)
    elif not os.path.isdir(sub_plot_dir):
        raise ValueError('File exists with the same name as the scion directory')
    elif OVERRIDE:
        # Delete directory even if it is not empty
        shutil.rmtree(sub_plot_dir)
        os.makedirs(sub_plot_dir)
    elif len(os.listdir(sub_plot_dir)) > 0:
        raise ValueError('Directory already exists')

    # Create subdirectories
    os.makedirs(os.path.join(sub_plot_dir, 'fc'))
    os.makedirs(os.path.join(sub_plot_dir, 'usage'))
    return sub_plot_dir


def fc_data_to_plot(source_dir, dest_dir):
    # TODO: Make single plot irec-scion
    fc_file = os.path.join(source_dir, 'fc.csv')
    if not os.path.exists(fc_file):
        return
    dtype = {
        'Isd-As1': str,
        'Isd-As2': str,
        'Duration': float,
    }
    fc_df = pd.read_csv(fc_file, dtype=dtype)
    # Get duration column as array
    duration = fc_df['Duration'].values
    # Create CDF plot
    duration = np.sort(duration)
    yvals = np.arange(len(duration)) / float(len(duration))
    plt.plot(duration, yvals)
    plt.xlabel('Duration (s)')
    plt.ylabel('Probability')
    #plt.title('Flow Completion Time')
    # Plot the 95th percentile
    percentile_95 = np.percentile(duration, 95)
    plt.axvline(x=percentile_95, color=colors[1], linestyle='--')
    plt.text(percentile_95+0.1, 0., f'95th percentile', rotation=90, color=colors[1])
    plt.savefig(os.path.join(dest_dir, 'fc_cdf.png'), bbox_inches='tight', dpi=300)
    plt.close()


def irec_data_to_plot(irec_name):
    sub_plot_dir = init_plot_dir(irec_name)
    irec_dir = os.path.join(data_dir, irec_name)

    # fc_path = os.path.join(irec_dir, 'fc')
    # if os.path.exists(fc_path):
    #     fc_data_to_plot(fc_path, os.path.join(sub_plot_dir, 'fc'))

    category_files_cs = {
        'Originate\nBeacon': 'originated_bcn.csv',
        'Propagation\nFiltering': 'prop_filter.csv',
        'Propagate\nBeacon': 'prop_bcn.csv',
        #'Handle\nBeacon': 'handle_bcn.csv',
        'Retrieve\nJob': 'job_retrieval.csv',
        'Mark\nBeacon': 'egress_mark.csv',
        'Store\nSegment': 'written.csv',
        'Fetch\nAlgorithm': 'algorithm.csv',
    }

    category_files_rac = {
        'Process\nJob': 'job_processing.csv',
        'Execute\nJob': 'job_execution.csv',
    }

    def create_box_plot(category_files, out_file):
        category_data = {}

        for category, file in category_files.items():
            file_path = os.path.join(irec_dir, file)
            if not os.path.exists(file_path):
                continue
            vals = pd.read_csv(file_path)['Total'].values
            if len(vals) == 0:
                continue
            category_data[category] = vals

        # Create box plot
        plt.boxplot(category_data.values(), tick_labels=category_data.keys(), showfliers=False, showmeans=True)
        plt.ylabel('Time (s)')
        plt.xticks(rotation=45)
        # Add text on top of each box
        for i, (category, vals) in enumerate(category_data.items()):
            # Calculate quantile without outliers
            q1 = np.percentile(vals, 75)
            # Add text on top of the box
            ms = q1 * 1000
            ms_str = "%.3fms" % ms
            plt.text(i+1, q1 * 1.01, ms_str, ha='center', va='bottom', color='black')
        plt.savefig(os.path.join(sub_plot_dir, out_file), bbox_inches='tight', dpi=300)
        plt.close()

    create_box_plot(category_files_cs, 'irec_cs.png')
    create_box_plot(category_files_rac, 'irec_rac.png')



if __name__ == '__main__':
    topology_dir = os.path.join(os.path.dirname(__file__), 'tier1')
    data_dir = os.path.join(topology_dir, 'data')
    if not os.path.exists(data_dir):
        raise ValueError('Data directory does not exist')
    plot_dir = os.path.join(topology_dir, 'plot')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    scion_dir_names = []
    irec_dir_names = []
    # Load all folders in the topology directory
    for dir in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, dir)):
            continue
        if dir.startswith('scion'):
            scion_dir_names.append(dir)
        elif dir.startswith('irec'):
            irec_dir_names.append(dir)

    # for scion_name in scion_dir_names:
    #     scion_to_data(scion_name)

    for irec_name in irec_dir_names:
        irec_data_to_plot(irec_name)

    pass
