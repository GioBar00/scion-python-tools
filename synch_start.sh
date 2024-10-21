
get_scenario_pods() {
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

get_binary_path_by_dev_type() {
  local dev_type=$1
  case $dev_type in
    "cs")
      echo "/app/control"
      ;;
    "rac")
      echo "/app/rac"
      ;;
    "sd")
      echo "/app/daemon"
      ;;
    *)
      echo "Error: Invalid device type."
      return 1
      ;;
  esac
}

get_executable_command_by_dev_type() {
  local dev_type=$1

  binary_path=$(get_binary_path_by_dev_type $dev_type)

  case $dev_type in
    "cs")
      echo "$binary_path --config /etc/scion/$dev_type.toml >> /var/log/startup.log 2>&1 &"
      ;;
    "rac")
      echo "$binary_path --config /etc/scion/$dev_type.toml >> /var/log/startup.log 2>&1 &"
      ;;
    "sd")
      echo "/etc/scion/full_conn.sh &"
      ;;
    *)
      echo "Error: Invalid device type."
      return 1
      ;;
  esac
}

label_selector="app=kathara"
namespace=$(kubectl get namespace -l=$label_selector)
if [ -z "$namespace" ]; then
    echo "Error: Kathara lab not running. Nothing to do."
    return 1
fi
namespace=$(kubectl get namespace -l=$label_selector -o=jsonpath='{.items[0].metadata.name}')
echo "Kathara lab running in namespace: $namespace"

# Compute date after 5m in seconds
start_time=$(($(date +%s)+30))

dev_types=("cs" "rac" "sd")

for dev_type in ${dev_types[@]}; do
    echo "Extracting scenario data for $dev_type devices..."
    dev_pods=$(get_scenario_pods $namespace $dev_type)
    for pod in ${dev_pods[@]}; do
        dev_name=$(get_device_name $pod)
        echo "Starting device: $dev_name"
        exec_command=$(get_executable_command_by_dev_type $dev_type)
        #echo kubectl exec -n $namespace $pod -- bash -c "\"sleep \$(($start_time - \$(date +%s)))s && $exec_command\""
        kubectl exec -n $namespace $pod -- bash -c "sleep \$(($start_time - \$(date +%s)))s && $exec_command" &
        PID=$!
        # Kill the process alter 10s
        (sleep 10 && kill -9 $PID) &
    done
done