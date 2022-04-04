set -euxo pipefail

RPC_HOST="172.16.2.241"
RPC_PORT="4445"
RPC_KEY="rtx-3080"

NUM_TRIALS=128


run () {
    name=$1
    target=$2
    device=$3
    tune_model=$4

    log_dir=$PWD/$name-$device
    mkdir -p $log_dir

    echo "Running model $name"
    python3 $TVM_HOME/tests/python/relax/e2e_autotir.py     \
        --model "$name"                                     \
        --target "$target"                                  \
        --device "$device"                                  \
        --num-trials $NUM_TRIALS                            \
        --work-dir "$log_dir"                               \
        --rpc-host "$RPC_HOST"                              \
        --rpc-port "$RPC_PORT"                              \
        --rpc-key "$RPC_KEY"                                \
        --tune-model $tune_model                            \
        2>&1 | tee "$log_dir/$name.log"
}


run resnet18 "llvm --num-cores=16" cpu 1
# run resnet18 "nvidia/geforce-rtx-2070" cuda 0
