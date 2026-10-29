#!/bin/bash
# Reproduce the four measured lanes (A, B2, B3, C) and log each one to results/.
#
#   bash run_lanes.sh setup        # once: write K.bin (2.05 GB) + V.bin (6.14 GB) to $DATA
#   bash run_lanes.sh A|B2|B3|C    # one lane
#   bash run_lanes.sh all          # A, B2, B3, C in order
#   bash run_lanes.sh D|Dhot       # InfiniteMemory lanes (card with an SSD, Computing mode):
#                                  #   D    = K from the card's SSD every recall (32 copies, cyclic)
#                                  #   Dhot = K already in the card's computable area
#                                  # D fills 32 x 8.2 GB = 262 GB on the card: run it on an idle card
#
#   DATA=/path/on/nvme  T=<MU cores>  DEV=0  bash run_lanes.sh all
#
# DATA must sit on the NVMe drive under test (a raw ext4/xfs partition), not on the
# XCENA card, so every lane reads K from the same place the original runs did.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="${DATA:-/mnt/data/bench_data}"
T="${T:-2816}"
DEV="${DEV:-0}"
PY="${PY:-python3}"
mkdir -p "$HERE/results"
stamp=$(date +%Y%m%d-%H%M%S)

run() {   # run <name> <command...>
    local log="$HERE/results/$1-$stamp.log"; shift
    echo "== $* " | tee "$log"
    "$@" 2>&1 | tee -a "$log"
}

lane() {
    case "$1" in
    setup) run setup $PY "$HERE/lanes/bench_retrieval_ab.py" --setup --data-dir "$DATA" ;;
    A)  run laneA  $PY "$HERE/lanes/bench_retrieval_ab.py" --lane A --data-dir "$DATA" -j 96 --io-threads 16 --direct ;;
    B2) run laneB2 $PY "$HERE/lanes/bench_lane_b.py"       --data-dir "$DATA" --io-threads 16 --direct ;;
    B3) KVIKIO_COMPAT_MODE=OFF run laneB3 $PY "$HERE/lanes/bench_lane_b_gds.py" --data-dir "$DATA" --io-threads 64 ;;
    C)  (cd "$HERE/dran_mu" && run laneC ./dran_retrieve_lanec -n 1000000 -t "$T" -b 3 --device "$DEV") ;;
    D)  (cd "$HERE/dran_mu" && run laneD ./im_tenant_bench --data "$DATA" --users 32 --slots 1 --order cyclic             --queries 64 --warmup 32 -t "$T" --device "$DEV") ;;
    Dhot) (cd "$HERE/dran_mu" && run laneDhot ./im_tenant_bench --data "$DATA" --users 1 --slots 1             --queries 40 --warmup 8 -t "$T" --device "$DEV") ;;
    *)  echo "unknown lane: $1"; exit 1 ;;
    esac
}

if [ "$1" = all ]; then for l in A B2 B3 C; do lane $l; done
else lane "${1:?usage: run_lanes.sh setup|A|B2|B3|C|D|Dhot|all}"; fi
