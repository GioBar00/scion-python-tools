import enum
import os.path
from pathlib import Path
from statistics import median

import pandas as pd
import shutil
import swifter
import numpy as np
import humanfriendly
from dask.array import minimum
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from cycler import cycler

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

    return sub_plot_dir


def fc_data_to_plot(source_dirs, dest_dir):
    init_plot_dir('fc')
    fc_data = {}
    for source_dir in source_dirs:
        fc_file = os.path.join(source_dir, 'fc', 'fc.csv')
        if not os.path.exists(fc_file):
            continue
        dtype = {
            'Isd-As1': str,
            'Isd-As2': str,
            'Duration': float,
        }
        key = source_dir.split('/')[-1]
        fc_data[key] = pd.read_csv(fc_file, dtype=dtype)['Duration'].values
        fc_data[key] = np.sort(fc_data[key])

    # Create CDF plot with all data
    plt.figure()
    for key, duration in fc_data.items():
        yvals = np.arange(len(duration)) / float(len(duration))
        yvals[-1] = 1
        plt.plot(duration, yvals, label=key.replace('scion', 'SCION').replace('irec-ubpf', 'IREC'))
    plt.xlabel(r'Duration \texttt{(s)}')
    plt.ylabel('CDF')
    plt.ylim(0, 1)
    # set xmin to 0
    plt.xlim(0, None)
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: r'\texttt{%.1f}' % x))
    plt.legend()
    #plt.title('Flow Completion Time')
    # for i, key in enumerate(fc_data.keys()):
    #     max = np.max(fc_data[key])
    #     plt.axvline(x=max, color=colors[i], linestyle='--')
    #     plt.text(np.max(fc_data[key])+0.1, 0, f'{max}s', rotation=90, color=colors[i], fontsize=8)
    # Plot the 95th percentile
    # percentile_95 = np.percentile(np.concatenate(list(fc_data.values())), 95)
    # plt.axvline(x=percentile_95, color=colors[-1], linestyle='--')
    # plt.text(percentile_95+0.1, 0., f'95th percentile', rotation=90, color=colors[1])
    # Add minor ticks
    plt.minorticks_on()
    # make grid transparent
    plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.savefig(os.path.join(dest_dir, 'fc', 'fc_cdf.pdf'), bbox_inches='tight', dpi=300)
    plt.close()


def usage_data_to_plot(usage_dirs, dest_dir):
    init_plot_dir('usage')
    usage_data = {}
    for usage_dir in usage_dirs:
        key = usage_dir.split('/')[-1]
        usage_data[key] = {}
        for t in ['cpu', 'mem', 'net_rx', 'net_tx']:
            usage_file = os.path.join(usage_dir, 'usage', f'{t}.csv')
            if not os.path.exists(usage_file):
                continue
            usage_data[key][t] = pd.read_csv(usage_file)
            usage_data[key][t]['Time'] = pd.to_datetime(usage_data[key][t]['Time'])
            t = t + '-pm'
            usage_file = os.path.join(usage_dir, 'usage', f'{t}.csv')
            if not os.path.exists(usage_file):
                continue
            usage_data[key][t] = {}
            df_pm = pd.read_csv(usage_file)
            for dev in DeviceType:
                df_dev = df_pm[df_pm['Dev Type'] == dev.value]
                if len(df_dev) == 0:
                    continue
                usage_data[key][t][dev] = {
                    'peak': np.sort(df_dev['Peak'].values),
                    'median': np.sort(df_dev['Median'].values),
                }

    def plot_usage(t, tpm, dev, k):
        plt.figure()
        median = []
        minimum = []
        for key, data in usage_data.items():
            if tpm not in data or dev not in data[tpm] or k not in data[tpm][dev]:
                continue
            # Plot cdf
            yvals = np.arange(len(data[tpm][dev][k])) / float(len(data[tpm][dev][k]))
            # Force last value to 1
            yvals[-1] = 1
            plt.plot(data[tpm][dev][k], yvals, label=key.replace('scion', 'SCION').replace('irec-ubpf', 'IREC'))
            median.append(np.median(data[tpm][dev][k]))
            minimum.append(np.min(data[tpm][dev][k]))
        if t == 'cpu':
            size = 1
            plt.xlabel(r'CPU (\%)')
            tick_interval = (plt.gca().get_xticks()[1] - plt.gca().get_xticks()[0]) * 100
            plt.gca().get_xaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'%.2f' % (x * 100) if tick_interval < 1 else f'%.0f' % (x * 100)))
        else:
            # Compute median memory unit
            unit = 'B'
            size = 1
            max_mem = np.max(median)
            if max_mem > 1e9:
                unit = 'GB'
                size = 1e9
            elif max_mem > 1e6:
                unit = 'MB'
                size = 1e6
            elif max_mem > 1e3:
                unit = 'KB'
                size = 1e3
            tick_interval = (plt.gca().get_xticks()[1] - plt.gca().get_xticks()[0]) / size
            plt.gca().get_xaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'%.2f' % (x / size) if tick_interval < 1 else f'%.0f' % (x / size)))
            if t == 'mem':
                plt.xlabel(r'Memory (%s)' % unit)
            elif t == 'net_rx':
                plt.xlabel(r'Network RX (%s/s)' % unit)
            elif t == 'net_tx':
                plt.xlabel(r'Network TX (%s/s)' % unit)

        # converted_min = [m / size for m in minimum]
        # if np.min(converted_min) < 2 * tick_interval:
        #     plt.xlim(0, None)
        # else:
        start = np.floor(np.min(minimum) / (size * tick_interval)) * tick_interval * size
        plt.xlim(start, None)
        #plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: r'\texttt{%.1f}' % x))
        #plt.ylabel(f'Probability {t} {k}')
        plt.ylabel(f'CDF')
        plt.ylim(0, 1)
        # plt.xscale('log')
        plt.legend()
        # Add minor ticks
        plt.minorticks_on()
        # make grid transparent
        plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.savefig(os.path.join(dest_dir, 'usage', f'{t}_{dev.value}_{k}.pdf'), bbox_inches='tight', dpi=300)
        plt.close()

    for t in ['cpu', 'mem', 'net_rx', 'net_tx']:
        tpm = t + '-pm'
        for dev in DeviceType:
            if dev == DeviceType.RAC:
                continue
            for k in ['peak', 'median']:
                plot_usage(t, tpm, dev, k)

        dev = DeviceType.RAC
        plt.figure()
        i = 0
        median = []
        minimum = []
        for key, data in usage_data.items():
            if t not in data or dev not in data[tpm] or 'peak' not in data[tpm][dev] or 'median' not in data[tpm][dev]:
                continue
            # Plot cdf
            yvals = np.arange(len(data[tpm][dev]['peak'])) / float(len(data[tpm][dev]['peak']))
            yvals[-1] = 1
            plt.plot(data[tpm][dev]['peak'], yvals, label=f'peak', color=colors[2])
            median.append(np.median(data[tpm][dev]['peak']))
            minimum.append(np.min(data[tpm][dev]['peak']))
            yvals = np.arange(len(data[tpm][dev]['median'])) / float(len(data[tpm][dev]['median']))
            yvals[-1] = 1
            plt.plot(data[tpm][dev]['median'], yvals, label=f'median', color=colors[2], linestyle='--')
            median.append(np.median(data[tpm][dev]['median']))
            minimum.append(np.min(data[tpm][dev]['median']))
            i += 1
        if t == 'cpu':
            size = 1
            plt.xlabel(r'CPU (\%)')
            tick_interval = (plt.gca().get_xticks()[1] - plt.gca().get_xticks()[0]) * 100
            # Humanize CPU
            plt.gca().get_xaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'%.2f' % (x * 100) if tick_interval < 1 else f'%.0f' % (x * 100)))
        else:
            unit = 'B'
            size = 1
            max_mem = np.max(median)
            if max_mem > 1e9:
                unit = 'GB'
                size = 1e9
            elif max_mem > 1e6:
                unit = 'MB'
                size = 1e6
            elif max_mem > 1e3:
                unit = 'KB'
                size = 1e3
            tick_interval = (plt.gca().get_xticks()[1] - plt.gca().get_xticks()[0]) / size
            plt.gca().get_xaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'%.2f' % (x / size) if tick_interval < 1 else f'%.0f' % (x / size)))
            if t == 'mem':
                plt.xlabel(r'Memory (%s)' % unit)
            elif t == 'net_rx':
                plt.xlabel(r'Network RX (%s/s)' % unit)
            elif t == 'net_tx':
                plt.xlabel(r'Network TX (%s/s)' % unit)

        # if np.min(minimum) < 2 * tick_interval:
        #     plt.xlim(0, None)
        start = np.floor(np.min(minimum) / (size * tick_interval)) * tick_interval * size
        plt.xlim(start, None)

        #plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: r'\texttt{%.1f}' % x))
        plt.ylabel(f'CDF')
        plt.ylim(0, 1)
        #plt.xscale('log')
        plt.legend()
        # Add minor ticks
        plt.minorticks_on()
        # make grid transparent
        plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.savefig(os.path.join(dest_dir, 'usage', f'{t}_{dev.value}.pdf'), bbox_inches='tight', dpi=300)
        plt.close()


def create_box_plot(source_dir, sub_plot_dir, category_files, out_file):
    category_data = {}

    for category, file in category_files.items():
        file_path = os.path.join(source_dir, file)
        if not os.path.exists(file_path):
            continue
        vals = pd.read_csv(file_path)['Total'].values
        if len(vals) == 0:
            continue
        category_data[category] = vals

    if len(category_data.items()) == 0:
        return
    # Create box plot
    bplot = plt.boxplot(category_data.values(), tick_labels=category_data.keys(), showfliers=False, showmeans=False, meanline=True,
                flierprops=dict(marker='x', markersize=1), patch_artist=True)
    # Set face color
    for patch in bplot['boxes']:
        patch.set_facecolor('white')
    plt.yscale('log')
    plt.ylabel('Time (s)')
    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    f.set_scientific(True)
    g = lambda x, pos: r"{\ttfamily $%s$}" % f.format_data(x)
    plt.gca().get_yaxis().set_major_formatter(g)
    #plt.xticks(rotation=45)
    # make grid transparent on y-axis
    plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    # Add text on top of each box
    for i, (category, vals) in enumerate(category_data.items()):
        # Calculate quantile without outliers
        q1 = np.percentile(vals, 75)
        # Add text on top of the box
        if q1 >= 1.:
            ms = q1
            ms_str = "%.2fs" % ms
        else:
            ms = q1 * 1000
            ms_str = "%.2fms" % ms
        #if i in [1, 5, 8, len(category_data.keys()) - 1]:
        if i in [len(category_data.keys()) - 1]:
            plt.text(i + 0.95, q1, ms_str, ha='right', va='bottom', color='black', fontsize=6)
        else:
            plt.text(i + 1.05, q1, ms_str, ha='left', va='bottom', color='black', fontsize=6)
    plt.savefig(os.path.join(sub_plot_dir, out_file), bbox_inches='tight', dpi=300)
    plt.close()


def scion_data_to_plot(scion_name):
    sub_plot_dir = init_plot_dir(scion_name)
    scion_dir = os.path.join(data_dir, scion_name)

    category_files_cs = {
        r'\begin{center}Retrieve\\Beacon\end{center}': 'prop_filter.csv',
        r'\begin{center}Store\\Segment\end{center}': 'written.csv',
        r'\begin{center}Originate\\Beacon\end{center}': 'originated_bcn.csv',
        r'\begin{center}Propagate\\Beacon\end{center}': 'prop_bcn.csv',
        r'\begin{center}Handle\\Beacon\end{center}': 'handle_bcn.csv',
    }

    create_box_plot(scion_dir, sub_plot_dir, category_files_cs, 'scion_cs.pdf')


def irec_data_to_plot(irec_name):
    sub_plot_dir = init_plot_dir(irec_name)
    irec_dir = os.path.join(data_dir, irec_name)

    category_files_cs = {
        #'Process\nJob': 'job_processing.csv',
        r'\begin{center}Retrieve\\Beacon\end{center}': 'job_retrieval.csv',
        r'\begin{center}Execute\\Beacon\end{center}': 'job_execution.csv',
        r'\begin{center}Filter\\Beacon\end{center}': 'prop_filter.csv',
        r'\begin{center}Mark\\Beacon\end{center}': 'egress_mark.csv',
        r'\begin{center}Store\\Segment\end{center}': 'written.csv',
        r'\begin{center}Fetch\\Algorithm\end{center}': 'algorithm.csv',
        r'\begin{center}Originate\\Beacon\end{center}': 'originated_bcn.csv',
        r'\begin{center}Propagate\\Beacon\end{center}': 'prop_bcn.csv',
        r'\begin{center}Handle\\Beacon\end{center}': 'handle_bcn.csv',
    }

    # category_files_rac = {
    # }

    create_box_plot(irec_dir, sub_plot_dir, category_files_cs, 'irec_cs.pdf')
    #create_box_plot(category_files_rac, 'irec_rac.png')



if __name__ == '__main__':
    topology_dir = os.path.join(os.path.dirname(__file__), 'tier1')
    data_dir = os.path.join(topology_dir, 'data')
    if not os.path.exists(data_dir):
        raise ValueError('Data directory does not exist')
    plot_dir = os.path.join(topology_dir, 'plot')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Font Times New Roman
    #plt.rcParams['font.family'] = 'Times New Roman'
    # Font size 9
    # plt.rcParams.update({'font.size': 9})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'cmr'
    plt.rcParams['text.usetex'] = True

    # Cycle through colors and line styles
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors[:4]) + cycler(linestyle=['-', '--', '-.', ':'])

    scion_dir_names = ['scion-30s', 'scion-20m']
    irec_dir_names = ['irec-ubpf']
    # Load all folders in the topology directory
    # for dir in os.listdir(data_dir):
    #     if not os.path.isdir(os.path.join(data_dir, dir)):
    #         continue
    #     if dir.startswith('scion'):
    #         scion_dir_names.append(dir)
    #     elif dir.startswith('irec'):
    #         irec_dir_names.append(dir)

    for scion_name in scion_dir_names:
        scion_data_to_plot(scion_name)

    for irec_name in irec_dir_names:
        irec_data_to_plot(irec_name)

    # Create fc plot
    #fc_data_to_plot([os.path.join(data_dir, scion_name) for scion_name in scion_dir_names] + [os.path.join(data_dir, irec_name) for irec_name in irec_dir_names], plot_dir)

    # Create usage plot
    #usage_data_to_plot([os.path.join(data_dir, scion_name) for scion_name in scion_dir_names] + [os.path.join(data_dir, irec_name) for irec_name in irec_dir_names], plot_dir)

    pass
