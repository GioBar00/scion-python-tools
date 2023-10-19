import glob
import os
import re
from typing import Dict, List

from kubernetes import client, config
from kubernetes.client import ApiException

NAMESPACE = "scion"
# Configs can be set in Configuration class directly or using helper utility
config.load_kube_config(config_file="jvanbommel-test-k8s-kubeconfig.yaml")

v1 = client.CoreV1Api()

def exists_namespace(namespace):
    namespaces = v1.list_namespace()
    for item in namespaces.items:
        if item.metadata.name == namespace:
            return True
    return False

def create_namespace(namespace):
    if exists_namespace(NAMESPACE):
        return
    metadata = client.V1ObjectMeta(
        name=namespace
    )
    v1.create_namespace(client.V1Namespace(metadata=metadata))


create_namespace(NAMESPACE)
def build_configmap_struct(path, namespace, all_volumes: List, dirs = []):
    files = {}
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            with open(os.path.join(path, file), 'r') as f:
                # create.setdefault(namespace, {})[file] = f.read()
                files[file] = f.read()
        else:
            new_dirs = dirs.copy()
            new_dirs.append(file)
            build_configmap_struct(os.path.join(path, file),
                                   (namespace + "." + file).replace("//", "_").replace("_", ".").lower(), all_volumes, new_dirs)

    all_volumes.append((namespace, dirs))
    print(f"Created configmap for {namespace}, dirs: {dirs}")
    metadata = client.V1ObjectMeta(
        name=namespace,
        namespace=NAMESPACE,
    )
    configmap = client.V1ConfigMap(
            api_version="v1",
            kind="ConfigMap",
            data=files,
            metadata=metadata
        )

    v1.create_namespaced_config_map(
        namespace=NAMESPACE,
        body=configmap,
        pretty='pretty_example',
    )

ex = []

v1.delete_collection_namespaced_config_map(namespace=NAMESPACE)
build_configmap_struct("gen/ASff00_0_110", "asff00.0.110.config", ex)
build_configmap_struct("gen/ASff00_0_111", "asff00.0.111.config", ex)
build_configmap_struct("gen/ASff00_0_112", "asff00.0.112.config", ex)

print("Built the config maps!\n\n")

print("The configuration entries are")
for namespace, dirs in ex:
    print(f"        - mountPath: /share/conf/{'/'.join(dirs)}\n          name: vol-{re.sub('[^0-9a-z]+', '-',namespace.lower())}")

for namespace, dirs in ex:
    print(f"      - name: vol-{re.sub('[^0-9a-z]+', '-',namespace.lower())}\n        configMap:\n          name: {namespace}")
