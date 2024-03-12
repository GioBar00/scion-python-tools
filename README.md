## Some useful commands:
export KUBECONFIG=

Create a topology
 ./topogen.py -c dockergen.topo -k --kube-push-config --kube-config kubeconfig.yaml --kube-ns scion

**Delete all deployments:**
`kubectl --kubeconfig=kubeconfig.yaml delete deployment --all -n=scion
`
**Deploy all generated:**
kubectl --kubeconfig=kubeconfig.yaml apply -f gen/kube/ --namespace=scion

**Redeploy all:**
kubectl --kubeconfig=kubeconfig.yaml rollout restart deployment -n scion

**View monitor:**
kubectl --kubeconfig=kubeconfig.yaml  apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml
```kubectl --kubeconfig=kubeconfig.yaml proxy```

http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/



Connect into two of the eh vms;
execute:
bin/scion address

bin/end2end --remote "1-ff00:00:112,10.244.0.134" --mode client

bin/scion address
bin/end2end --local "1-ff00:00:112,10.244.0.134" --mode server


Check border router logs => it is actually getting routed via scion.

while :; do bin/end2end --remote "1-ff00:00:112,10.244.0.134:1025" --local "1-ff00:00:111,10.244.0.189" --mode client; done;


