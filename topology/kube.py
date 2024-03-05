import os
import re
from typing import List

import yaml
from kubernetes import client, config

from topology.defines import KUBE_GEN_PATH
from topology.util import write_file


class KubernetesConfigMapGenerator(object):
    """
    Converts the created configuration to configmaps and pushes these to the provided Kubernetes cluster.
    """

    def __init__(self, args):
        """
        Initialize an instance of the class ConfigGenerator.

        :param KubernetesConfigMapGeneratorArgs args: Contains the passed command line arguments.
        """
        self.args = args
        # Load the kubernetes config
        config.load_kube_config(config_file=self.args.kube_config)
        # And create an API instance for kubernetes
        self.client = client.CoreV1Api()
        self.kube_path = os.path.join(self.args.output_dir, KUBE_GEN_PATH)

    def generate_configmaps(self, topo_dicts):
        self._create_namespace(self.args.kube_ns)
        self.client.delete_collection_namespaced_config_map(namespace=self.args.kube_ns)
        for topo_id, topo in topo_dicts.items():
            base = topo_id.base_dir(self.args.output_dir)
            # print(base, topo_id.kube_name())
            generated = list()
            self._add_configmap_struct(base, topo_id.kube_name(), generated)
            self.generate_br_deployments(topo_id, topo, generated)
            self.generate_cs_deployments(topo_id, topo, generated)
            self.generate_eh_deployments(topo_id, topo, generated)
            # for k, v in topo.get("border_routers", {}).items():
            #     base = topo_id.base_dir(self.args.output_dir)

    def generate_br_deployments(self, topo_id, topo, generated_maps):
        br_ports = [50000, 30042, 30043, 30044, 30045, 30046, 30047, 30048, 30049, 30050, 30051]
        allowed_dirs = ['prometheus', 'crypto/as', 'crypto/ca', 'crypto/voting', 'crypto', 'keys', 'certs', '']
        for k, v in topo.get("border_routers", {}).items():
            dep_name = re.sub('[^0-9a-z]+', '-',k)
            br = {"apiVersion": "apps/v1",
                  "kind": "Deployment",
                  "metadata": {
                      "name": dep_name,
                      "labels": {
                          "service": dep_name
                      }
                  },
                  "spec": {
                      "replicas": 1,
                      "selector": {
                          "matchLabels": {
                              "service": dep_name
                          }
                      },
                      "strategy": {
                          "type": "Recreate"
                      },
                      "template": {
                          "metadata": {
                              "labels": {
                                  "service": dep_name
                                  #     TODO: Link
                              }
                          },
                          "spec": {
                              "securityContext": {
                                  "runAsUser": 1000,
                                  "runAsGroup": 1000,
                              },
                              "initContainers": [{
                                  "name": "init",
                                  # Initcontainers so the pods already get cluster IPs assigned prior to SCION starting.
                                  "image": "busybox:1.28",
                                  "command": ["sh", "-c", "sleep 5"]
                              }],
                              "containers": [{
                                  "args": ["--config", "/share/conf/" + k + ".toml"],
                                  "env": [
                                      {"name": "SCION_EXPERIMENTAL_BFD_DESIRED_MIN_TX"},
                                      {"name": "SCION_EXPERIMENTAL_BFD_DETECT_MULT"},
                                      {"name": "SCION_EXPERIMENTAL_BFD_REQUIRED_MIN_RX"},
                                  ],
                                  "image": "registry.digitalocean.com/scion-on-kubernetes/posix-router:latest",
                                  "name": dep_name,
                                  "volumeMounts": [],
                              }],
                              "restartPolicy": "Always",
                              "volumes": []
                          }
                      }
                  }
                  }
            br_service = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": dep_name + "-svc",
                },
                "spec": {
                    "clusterIP": "None",
                    "publishNotReadyAddresses": True,
                    "selector": {
                        "service": dep_name
                    },
                    "ports": []
                },
            }
            for name, dirs in generated_maps:
                directory = '/'.join(dirs)
                if directory not in allowed_dirs:
                    continue
                br["spec"]["template"]["spec"]["volumes"].append({
                    "name": "vol-" + re.sub('[^0-9a-z]+', '-', name),
                    "configMap": {
                        "name": name
                    }
                })
                br["spec"]["template"]["spec"]["containers"][0]["volumeMounts"].append({
                    "name": "vol-" + re.sub('[^0-9a-z]+', '-', name),
                    "mountPath": "/share/conf/" + directory,
                    "readOnly": True
                })
            for index, port in enumerate(br_ports):
                br_service["spec"]["ports"].append({"name": f"port{index}", "port": port, "protocol": "UDP",
                                                    "targetPort": port})
            config_path = os.path.join(self.kube_path, f"{dep_name}.yaml")
            write_file(config_path, yaml.dump_all([br, br_service], default_flow_style=False))
    def generate_cs_deployments(self, topo_id, topo, generated_maps):
        cs_ports = [30252, 30041, 30033, 30034, 30035]
        allowed_dirs = ['prometheus', 'crypto/as', 'crypto', 'keys', 'certs', '']
        for k, v in topo.get("control_service", {}).items():
            dep_name = re.sub('[^0-9a-z]+', '-',k)
            cs = {"apiVersion": "apps/v1",
                  "kind": "Deployment",
                  "metadata": {
                      "name": dep_name,
                      "labels": {
                          "service": dep_name
                      }
                  },
                  "spec": {
                      "replicas": 1,
                      "selector": {
                          "matchLabels": {
                              "service": dep_name
                          }
                      },
                      "strategy": {
                          "type": "Recreate"
                      },
                      "template": {
                          "metadata": {
                              "labels": {
                                  "service": dep_name
                                  #     TODO: Links
                              }
                          },
                          "spec": {
                              "initContainers": [{
                                  "name": "init",
                                  # Initcontainers so the pods already get cluster IPs assigned prior to SCION starting.
                                  "image": "busybox:1.28",
                                  "command": ["sh", "-c", "sleep 5"]
                              }],
                              "containers": [{
                                  "args": ["--config", "/share/conf/" + k + ".toml"],
                                  "image": "registry.digitalocean.com/scion-on-kubernetes/control:latest",
                                  "name": dep_name,
                                  "volumeMounts": [],  # todo
                              }, {
                                  "args": ["--config", "/share/conf/disp_" + k + ".toml"],
                                  "image": "registry.digitalocean.com/scion-on-kubernetes/dispatcher:latest",
                                  "name": "disp-" + dep_name,
                                  "volumeMounts": [],
                              }
                              ],
                              "restartPolicy": "Always",
                              "volumes": []
                          }
                      }
                  }
                  }
            cs_service = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": dep_name + "-svc",
                },
                "spec": {
                    "clusterIP": "None",
                    "publishNotReadyAddresses": True,
                    "selector": {
                        "service": dep_name
                    },
                    "ports": []
                },
            }

            for name, dirs in generated_maps:
                directory = '/'.join(dirs)
                if directory not in allowed_dirs:
                    continue
                cs["spec"]["template"]["spec"]["volumes"].append({
                    "name": "vol-" + re.sub('[^0-9a-z]+', '-', name),
                    "configMap": {
                        "name": name
                    }
                })
                for container in cs["spec"]["template"]["spec"]["containers"]:
                    container["volumeMounts"].append({
                        "name": "vol-" + re.sub('[^0-9a-z]+', '-', name),
                        "mountPath": "/share/conf/" + directory,
                        "readOnly": True
                    })
            for index, port in enumerate(cs_ports):
                cs_service["spec"]["ports"].append({"name": f"port{index}", "port": port, "protocol": "UDP",
                                                    "targetPort": port})

            config_path = os.path.join(self.kube_path, f"{dep_name}.yaml")
            write_file(config_path, yaml.dump_all([cs, cs_service], default_flow_style=False))
    def generate_eh_deployments(self, topo_id, topo, generated_maps):
        allowed_dirs = ['prometheus', 'crypto/as', 'crypto', 'keys', 'certs', '']
        dep_name = re.sub('[^0-9a-z]+', '-',topo_id.file_fmt().lower())
        eh = {"apiVersion": "apps/v1",
              "kind": "Deployment",
              "metadata": {
                  "name": "eh" + dep_name,
                  "labels": {
                      "service": "eh" + dep_name
                  }
              },
              "spec": {
                  "replicas": 1,
                  "selector": {
                      "matchLabels": {
                          "service": "eh" + dep_name
                      }
                  },
                  "strategy": {
                      "type": "Recreate"
                  },
                  "template": {
                      "metadata": {
                          "labels": {
                              "service": "eh" + dep_name
                              #  TODO: Link
                          }
                      },
                      "spec": {
                          "initContainers": [{
                              "name": "init",
                              # Initcontainers so the pods already get cluster IPs assigned prior to SCION starting.
                              "image": "busybox:1.28",
                              "command": ["sh", "-c", "sleep 5"]
                          }],
                          "containers": [{
                              "args": ["--config", "/share/conf/sd.toml"],
                              "image": "registry.digitalocean.com/scion-on-kubernetes/endhost:latest",
                              "name": "eh" + dep_name + "-eh",
                              "volumeMounts": [],
                          }],
                          "restartPolicy": "Always",
                          "volumes": []
                      }
                  }
              }
              }

        for name, dirs in generated_maps:
            directory = '/'.join(dirs)
            if directory not in allowed_dirs:
                continue
            eh["spec"]["template"]["spec"]["volumes"].append({
                "name": "vol-" + re.sub('[^0-9a-z]+', '-', name),
                "configMap": {
                    "name": name
                }
            })
            eh["spec"]["template"]["spec"]["containers"][0]["volumeMounts"].append({
                "name": "vol-" + re.sub('[^0-9a-z]+', '-', name),
                "mountPath": "/share/conf/" + directory,
                "readOnly": True
            })

            config_path = os.path.join(self.kube_path, f"eh{dep_name}.yaml")
            write_file(config_path, yaml.dump(eh, default_flow_style=False))


    def _add_configmap_struct(self, path, name, all_volumes: List, dirs=[]):
        files = {}
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                with open(os.path.join(path, file), 'r') as f:
                    files[file] = f.read()
            else:
                new_dirs = dirs.copy()
                new_dirs.append(file)
                self._add_configmap_struct(os.path.join(path, file),
                                           (name + "." + file).replace("//", "_").replace("_", ".").lower(),
                                           all_volumes, new_dirs)
        all_volumes.append((name, dirs))
        print(f"Created configmap for {name}, dirs: {dirs}")
        metadata = client.V1ObjectMeta(
            name=name,
            namespace=self.args.kube_ns,
        )
        configmap = client.V1ConfigMap(
            api_version="v1",
            kind="ConfigMap",
            data=files,
            metadata=metadata
        )

        self.client.create_namespaced_config_map(
            namespace=self.args.kube_ns,
            body=configmap,
            pretty='pretty_example',
        )

    def _exists_namespace(self, namespace):
        namespaces = self.client.list_namespace()
        for item in namespaces.items:
            if item.metadata.name == namespace:
                return True
        return False

    def _create_namespace(self, namespace):
        if self._exists_namespace(namespace):
            return
        metadata = client.V1ObjectMeta(name=namespace)
        self.client.create_namespace(client.V1Namespace(metadata=metadata))
