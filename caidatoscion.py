import math
import shutil
import string
import xml.etree.ElementTree as ET
from typing import Dict

import yaml

from topology.config import ConfigGenerator

IREC = True


def parseNode(child):
    node = {}
    if not hasattr(child, "get"):
        print(f" ERR! parseNode called with invalid XML child")
        return None
    if child.get('id.type') == 'int':
        node['id'] = child.get('id')
    if len(child) < 5:
        print(" ERR! not enough properties in XML node.")

    for property in child:
        key = property.attrib['name']
        val = property.text
        if property.attrib['type'] == 'int':
            node[key] = int(val)
        elif property.attrib['type'] == 'float':
            node[key] = float(val)
    return node


def parseLink(child):
    link = {}
    if not hasattr(child, "get"):
        print(f" ERR! parseLink called with invalid XML child")
        return None
    if len(child) < 8:
        print(" ERR! not enough properties in XML node.")

    for property in child:
        if property.tag == "from" and property.attrib['type'] == 'int':
            link['from'] = int(property.text)
        elif property.tag == "to" and property.attrib['type'] == 'int':
            link['to'] = int(property.text)

        elif property.tag == "property":
            key = property.attrib['name']
            val = property.text
            if property.attrib['type'] == 'int':
                link[key] = int(val)
            elif property.attrib['type'] == 'float':
                link[key] = float(val)
            elif property.attrib['type'] == 'string':
                link[key] = str(val)
        else:
            print(f" ERR! could not process property {property.tag} ({property.attrib})")
    return link


# def parse

def parseXML(file: str) -> [Dict, Dict]:
    tree = ET.parse(file)
    root = tree.getroot()
    nodes = []
    links = []
    for child in root:
        if child.tag == "node":
            nodes.append(parseNode(child))
        elif child.tag == "link":
            links.append(parseLink(child))
        else:
            print(f" ERR! unknown tag; {child.tag}")
    return nodes, links


MAX_PORTS_PER_BR = 25


class KubernetesGenerator:
    def __init__(self):
        self.asn_counter = 1
        self.asn_mapping = {}

    def asn(self, as_id):  # TODO(jvanbommel): support different ISDs
        if as_id not in self.asn_mapping:
            self.asn_mapping[as_id] = self.asn_counter
            self.asn_counter += 1
        return f'1-ff00:0:{self.asn_mapping[as_id]}'


def main():
    # shutil.rmtree('gen2')
    kgen = KubernetesGenerator()
    nodes, links = parseXML("tier1.xml")
    sorted_nodes = sorted(nodes, key=lambda x: x['interface_count'], reverse=True)

    num_links_pair = {}

    core_ases = sorted_nodes[:50]
    core_ids = [int(as_['id']) for as_ in core_ases]
    print("Core ASes are: " + ', '.join([as_['id'] for as_ in core_ases]))
    non_core_ases = sorted_nodes[50:]

    print(f"There are {len(non_core_ases)} non-Core ASes")

    # with open("default.topo") as f:
    #     print(yaml.load(f, Loader=yaml.SafeLoader))

    as_intf_state = {int(x['id']): 0 for x in nodes}
    MAX_LINKS_PER_PAIR = 50

    as_dict = {'ASes': {}, 'links': []}
    for core_as in core_ids:
        as_name = kgen.asn(core_as)

        as_dict['ASes'][as_name] = {'core': True, 'voting': True, 'authoritative': True, 'issuing': True, "dispatched_ports": "1024-65535",
                                    'dynamic_racs': 1,
                                    'irec': {'algorithms': [{'originate': True, 'id': 0, 'file': 'algorithms/deadbeef.wasm'}]}
                                    }
    for link in links:
        if link['from'] in core_ids and link['to'] in core_ids:
            link_id = [link['from'], link['to']]
            link_id.sort()
            link_id = str(link_id)
            if link_id in num_links_pair and num_links_pair[link_id] >= MAX_LINKS_PER_PAIR:
                continue
                
            # portA = '-' + string.ascii_uppercase[
            #     math.floor(as_intf_state[link['from']] / MAX_PORTS_PER_BR)] + '#' + str(as_intf_state[link['from']])
            # portB = '-' + string.ascii_uppercase[math.floor(as_intf_state[link['to']] / MAX_PORTS_PER_BR)] + '#' + \
            #         str(as_intf_state[link['to']])

            portA = '-' + str(
                math.floor(as_intf_state[link['from']] / MAX_PORTS_PER_BR)) + '#' + str(as_intf_state[link['from']])
            portB = '-' + str(math.floor(as_intf_state[link['to']] / MAX_PORTS_PER_BR)) + '#' + \
                    str(as_intf_state[link['to']])

            as_dict['links'].append(
                {'a': f"{kgen.asn(link['from'])}{portA}", 'b': f"{kgen.asn(link['to'])}{portB}", 'linkAtoB': 'CORE'})
            as_intf_state[link['from']] += 1
            as_intf_state[link['to']] += 1
            if link_id in num_links_pair:
                num_links_pair[link_id] += 1
            else:
                num_links_pair[link_id] = 1
        #    TODO(jvanbommel)     In CAIDA almost all links are link['rel'] == 'peer'...
        else:
            print(f"ERR! Did not create link for {link} ")
    for non_core_as in non_core_ases:
        as_name = f'1-ff00:0:{non_core_as["id"]}'
        as_dict['ASes'].append({as_name: {'cert_issuer': '1-ff00:0:120', "dispatched_ports": "1024-65535"}})  # TODO(jvanbommel); configurable.
    with open('tier1-rac-wa-no_disp.topo', 'w') as outfile:
        yaml.dump(as_dict, outfile)
    #
    # class DockerConfig:
    #     pass
    #
    # config = DockerConfig()
    # config.skip_parsing = True
    # config.docker = True
    # config.network = None
    # config.random_ifids = True
    # config.sig = False
    # config.output_dir = "gen2"
    # config.topo_config = "dockergen.topo"
    # config.topo_config_preloaded = as_dict
    # config.features = []
    # config.docker_registry = ''
    # config.image_tag = ''
    #
    # confgen = ConfigGenerator(config)
    # confgen.generate_all()


if __name__ == '__main__':
    main()
