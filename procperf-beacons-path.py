import os
from pathlib import Path
import enum
import pandas as pd
from collections import defaultdict

from pandas import DataFrame


class ActionType(enum.Enum):
    Received = 0
    Originated = 1
    Propagated = 2
    Processed = 3


def build_packet_tree(df: DataFrame, packet_id: str, current_device: str = None):
    tree = defaultdict(list)

    rows = df[df['ID'] == packet_id]

    if current_device is None:
        # current device is the one that Originated the packet
        opt_dev = rows[rows['Type'] == ActionType.Originated.name]['Device'].values
        if len(opt_dev) == 0:
            # current device is the one that Propagated the packet (packet_id == 'Next ID')
            opt_dev = df[df['Next ID'] == packet_id and df['Type'] == ActionType.Propagated.name]['Device'].values

        if len(opt_dev) == 0:
            raise ValueError(f"Packet ID {packet_id} not found in the dataframe.")
        current_device = opt_dev[0]

        start_dev = current_device
    else:
        start_dev = None

    rows = rows[rows['Device'] != current_device]

    # For each propagation of this packet, record the next devices
    for _, row in rows.iterrows():
        if pd.notnull(row['Next ID']) and row['Type'] != ActionType.Originated.name:
            next_id = row['Next ID']
            next_device = row['Device']

            # Add this device and next packet ID to the current tree branch
            tree[current_device].append((next_device, next_id))

            # Recursively trace the next packet ID
            subtree, _ = build_packet_tree(df, next_id, next_device)

            # Merge the subtree into the main tree
            for sub_device, sub_path in subtree.items():
                tree[sub_device].extend(sub_path)

    return tree, start_dev


def print_packet_tree(tree, current_device=None, level=0, printed_devs=None):
    """
    Recursively prints the packet propagation tree in a hierarchical format.

    Parameters:
    - tree: The tree structure containing the propagation path.
    - current_device: The device currently handling the packet.
    - level: The indentation level for pretty printing (used for recursion).
    """

    if printed_devs is None:
        printed_devs = []
    if current_device not in printed_devs:
        print("    " * level + f"Device: {current_device}")

    # Traverse all next hops (propagations) from the current device
    for device, next_id in tree[current_device]:
        # print("    " * (level + 1) + f"Propagated to {device}, Packet ID: {next_id}")
        print_packet_tree(tree, device, level + 1, printed_devs)
        if device not in printed_devs:
            printed_devs.append(device)

def print_span_from_as(df: DataFrame, as_id: str):
    cs = f"cs{as_id.replace('-', '_').replace(':', '_')}_1"
    ids = df[(df['Type'] == ActionType.Originated.name) & (df['Device'] == cs)]['ID'].values
    tree = defaultdict(list)
    for id in ids:
        subtree, _ = build_packet_tree(df, id, cs)
        for device, path in subtree.items():
            tree[device].extend(path)

    print_packet_tree(tree, cs)

if __name__ == '__main__':
    scion_dir = os.path.join(os.getcwd(), f"procperf-tier1-irec-ubpf-nopath/")

    scion_devs = [Path(f).stem for f in os.listdir(scion_dir) if f.endswith(".csv")]

    # Load data from csv
    # df = pd.read_csv(os.path.join(scion_dir, f"{scion_cses[0]}.csv"), sep=";")
    dfs_dev = [pd.read_csv(os.path.join(scion_dir, f"{dev}.csv"), sep=";") for dev in scion_devs]
    # Add column with the name of the csv file
    for i in range(len(dfs_dev)):
        dfs_dev[i]["Device"] = scion_devs[i]
    # Concatenate all dataframes into one
    df = pd.concat(dfs_dev)

    # beacon_id = "b2e2d6d6804e6bacf5547972 99e0"
    # tree, start_dev = build_packet_tree(df, beacon_id)
    # print_packet_tree(tree, start_dev)

    # print_span_from_as(df, "1-ff00:0:1")

    # Beacons that have been received but not processed
    df_received = df[df['Type'] == ActionType.Received.name]
    df_processed = df[df['Type'] == ActionType.Processed.name]
    df_not_processed = df_received[~df_received['ID'].isin(df_processed['ID'])]
    print("Number of beacons received but not processed: ", len(df_not_processed))


