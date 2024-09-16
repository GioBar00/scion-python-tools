
get_procperf_pods() {
    local namespace=$1
    local dev_type=$2
    if [ -z "$dev_type" ]; then
        echo "Error: Device type not provided."
        return 1
    fi
    if [ -z "$namespace" ]; then
        echo "Error: Namespace not provided."
        return 1
    fi
    
    pods=$(kubectl get pods -n "$namespace" -l app=kathara -o=jsonpath='{.items[*].metadata.name}')

    # Filter pod names that start with kathara-dev_type
    dev_pods=()
    for pod in $pods; do
        if [[ $pod == kathara-$dev_type* ]]; then
            dev_pods+=($pod)
        fi
    done
    echo ${dev_pods[@]}
}

get_hostname_from_pod() {
    local pod=$1
    if [ -z "$pod" ]; then
        echo "Error: Pod name not provided."
        return 1
    fi

    # Split pod name by - and ignore last 2 parts
    hostname=$(echo $pod | awk -F'-' '{for(i=1;i<NF-1;++i) printf "%s-", $i; print ""}')
    # remove trailing -
    hostname=${hostname%-}
    echo $hostname
}

get_procperf_file_path() {
    local pod=$1
    if [ -z "$pod" ]; then
        echo "Error: Pod name not provided."
        return 1
    fi

    hostname=$(get_hostname_from_pod $pod)
    echo /share/procperf-$hostname.csv
}

get_device_name() {
    local pod=$1
    if [ -z "$pod" ]; then
        echo "Error: Pod name not provided."
        return 1
    fi

    # Split pod name by - and ignore first and last 3 parts
    dev_name=$(echo $pod | awk -F'-' '{for(i=2;i<NF-2;++i) printf "%s_", $i; print ""}')
    # remove trailing _
    dev_name=${dev_name%_}
    echo $dev_name
}

label_selector="app=kathara"
namespace=$(kubectl get namespace -l=$label_selector)
if [ -z "$namespace" ]; then
    echo "Error: Kathara lab not running. Nothing to do."
    return 1
fi
namespace=$(kubectl get namespace -l=$label_selector -o=jsonpath='{.items[0].metadata.name}')
echo "Kathara lab running in namespace: $namespace"

# Check if destination directory exists
dest_dir="./procperf"
if [ ! -d "$dest_dir" ]; then
    mkdir -p $dest_dir
fi

dev_types=("cs" "rac")

for dev_type in ${dev_types[@]}; do
    echo "Extracting procperf data for $dev_type devices..."
    dev_pods=$(get_procperf_pods $namespace $dev_type)
    for pod in ${dev_pods[@]}; do
        echo "Extracting procperf data for pod: $pod"
        procperf_file=$(get_procperf_file_path $pod)
        dev_name=$(get_device_name $pod)
        kubectl exec -n $namespace $pod -- cp $procperf_file $dev_name.csv
        kubectl cp --retries 5 $namespace/$pod:$dev_name.csv $dest_dir/$dev_name.csv
        kubectl exec -n $namespace $pod -- rm $dev_name.csv
    done
done