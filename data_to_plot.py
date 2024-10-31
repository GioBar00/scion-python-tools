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
    for _, duration in fc_data.items():
        yvals = np.arange(len(duration)) / float(len(duration))
        plt.plot(duration, yvals)
    plt.xlabel('Duration (s)')
    plt.ylabel('Probability')
    plt.legend(fc_data.keys())
    #plt.title('Flow Completion Time')
    for i, key in enumerate(fc_data.keys()):
        max = np.max(fc_data[key])
        plt.axvline(x=max, color=colors[i], linestyle='--')
        plt.text(np.max(fc_data[key])+0.1, 0, f'{max}s', rotation=90, color=colors[i], fontsize=8)
    # Plot the 95th percentile
    # percentile_95 = np.percentile(np.concatenate(list(fc_data.values())), 95)
    # plt.axvline(x=percentile_95, color=colors[-1], linestyle='--')
    # plt.text(percentile_95+0.1, 0., f'95th percentile', rotation=90, color=colors[1])
    plt.savefig(os.path.join(dest_dir, 'fc', 'fc_cdf.png'), bbox_inches='tight', dpi=300)
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
        for key, data in usage_data.items():
            if tpm not in data or dev not in data[tpm] or k not in data[tpm][dev]:
                continue
            # Plot cdf
            yvals = np.arange(len(data[tpm][dev][k])) / float(len(data[tpm][dev][k]))
            plt.plot(data[tpm][dev][k], yvals, label=key)
        if t == 'cpu':
            plt.xlabel('CPU')
            # Humanize CPU
            plt.gca().get_xaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'%.1fm' % (x * 1000) if x < 1 else f'%.1f' % x))
        elif t == 'mem':
            plt.xlabel('Memory')
            # Humanize memory
            plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: humanfriendly.format_size(x)))
        elif t == 'net_rx':
            plt.xlabel('Network RX')
            # Humanize network
            plt.gca().get_xaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, _: humanfriendly.format_size(x) + '/s'))
        elif t == 'net_tx':
            plt.xlabel('Network TX')
            # Humanize network
            plt.gca().get_xaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, _: humanfriendly.format_size(x) + '/s'))
        plt.ylabel(f'Probability {t} {k}')
        # plt.xscale('log')
        plt.legend()
        plt.savefig(os.path.join(dest_dir, 'usage', f'{t}_{dev.value}_{k}.png'), bbox_inches='tight', dpi=300)
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
        for key, data in usage_data.items():
            if t not in data or dev not in data[tpm] or 'peak' not in data[tpm][dev] or 'median' not in data[tpm][dev]:
                continue
            # Plot cdf
            yvals = np.arange(len(data[tpm][dev]['peak'])) / float(len(data[tpm][dev]['peak']))
            plt.plot(data[tpm][dev]['peak'], yvals, label=f'peak', color=colors[i])
            yvals = np.arange(len(data[tpm][dev]['median'])) / float(len(data[tpm][dev]['median']))
            plt.plot(data[tpm][dev]['median'], yvals, label=f'median', color=colors[i], linestyle='--')
            i += 1
        if t == 'cpu':
            plt.xlabel('CPU')
            # Humanize CPU
            plt.gca().get_xaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'%.1fm' % (x * 1000) if x < 1 else f'%.1f' % x))
        elif t == 'mem':
            plt.xlabel('Memory')
            # Humanize memory
            plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: humanfriendly.format_size(x)))
        elif t == 'net_rx':
            plt.xlabel('Network RX')
            # Humanize network
            plt.gca().get_xaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, _: humanfriendly.format_size(x) + '/s'))
        elif t == 'net_tx':
            plt.xlabel('Network TX')
            # Humanize network
            plt.gca().get_xaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, _: humanfriendly.format_size(x) + '/s'))
        plt.ylabel(f'Probability {t}')
        #plt.xscale('log')
        plt.legend()
        plt.savefig(os.path.join(dest_dir, 'usage', f'{t}_{dev.value}.png'), bbox_inches='tight', dpi=300)
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
    plt.boxplot(category_data.values(), tick_labels=category_data.keys(), showfliers=False, showmeans=True, meanline=True, flierprops=dict(marker='x', markersize=1))
    plt.yscale('log')
    plt.ylabel('Time (s)')
    plt.xticks(rotation=45)
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
        if i in [1, 5, 8, len(category_data.keys()) - 1]:
            plt.text(i + 0.95, q1, ms_str, ha='right', va='bottom', color='black', fontsize=6)
        else:
            plt.text(i + 1.05, q1, ms_str, ha='left', va='bottom', color='black', fontsize=6)
    plt.savefig(os.path.join(sub_plot_dir, out_file), bbox_inches='tight', dpi=300)
    plt.close()


def scion_data_to_plot(scion_name):
    sub_plot_dir = init_plot_dir(scion_name)
    scion_dir = os.path.join(data_dir, scion_name)

    category_files_cs = {
        'Propagation\nFiltering': 'prop_filter.csv',
        'Store\nSegment': 'written.csv',
        'Originate\nBeacon': 'originated_bcn.csv',
        'Propagate\nBeacon': 'prop_bcn.csv',
        'Handle\nBeacon': 'handle_bcn.csv',
    }

    create_box_plot(scion_dir, sub_plot_dir, category_files_cs, 'scion_cs.png')


def irec_data_to_plot(irec_name):
    sub_plot_dir = init_plot_dir(irec_name)
    irec_dir = os.path.join(data_dir, irec_name)

    category_files_cs = {
        'Process\nJob': 'job_processing.csv',
        'Retrieve\nJob': 'job_retrieval.csv',
        'Execute\nJob': 'job_execution.csv',
        'Propagation\nFiltering': 'prop_filter.csv',
        'Mark\nBeacon': 'egress_mark.csv',
        'Store\nSegment': 'written.csv',
        'Fetch\nAlgorithm': 'algorithm.csv',
        'Originate\nBeacon': 'originated_bcn.csv',
        'Propagate\nBeacon': 'prop_bcn.csv',
        'Handle\nBeacon': 'handle_bcn.csv',
    }

    # category_files_rac = {
    # }

    create_box_plot(irec_dir, sub_plot_dir, category_files_cs, 'irec_cs.png')
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
    fc_data_to_plot([os.path.join(data_dir, scion_name) for scion_name in scion_dir_names] + [os.path.join(data_dir, irec_name) for irec_name in irec_dir_names], plot_dir)

    # Create usage plot
    usage_data_to_plot([os.path.join(data_dir, scion_name) for scion_name in scion_dir_names] + [os.path.join(data_dir, irec_name) for irec_name in irec_dir_names], plot_dir)

    pass
