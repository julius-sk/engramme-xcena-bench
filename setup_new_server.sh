#!/usr/bin/env bash
# setup_new_server.sh — bring a bare Ubuntu 24.04 server with an XCENA MX1P up to the
# state the reference server was in when lanes A / B2 / B3 / C were measured.
#
# Run the stages in order. Stages that need a reboot say so and stop.
#
#   bash setup_new_server.sh check                    # 0  read-only: OS, kernel, devices, network
#   bash setup_new_server.sh apt                      # 1  system packages (+ kernel if too old)
#   bash setup_new_server.sh sdk  <sdk.tar.xz>        # 2  XCENA SDK (match the card fw: 1.0.10->v1.4.9, 1.0.12->v1.4.11) + mx_dma
#   bash setup_new_server.sh compute                  # 3  InfiniteMemory card: turn Computing mode on (then reboot)
#   bash setup_new_server.sh fw   <slot>              #    (rarely needed) flash firmware; never older than running
#                                                     #    -> then: sudo shutdown -h now (cold boot)
#   DAX="daxA.0 daxB.0" bash setup_new_server.sh dax  # 4  devdax + pxl_resourced (every boot)
#   bash setup_new_server.sh validate                 # 5  official validation, Computable, MU cores
#   bash setup_new_server.sh gpu  [deb_dir]           # 6  only if an A100 is present: open driver + GDS
#   bash setup_new_server.sh py                       # 7  python venv for lanes A / B2 / B3
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
# Only for a firmware update XCENA has approved for this card; Computing mode itself is the `compute` stage.
FW="${FW:-$HERE/env/firmware/firmware.bin}"
VENV="${VENV:-$HOME/venv-torch}"
DEV="${DEV:-0}"

ok()   { echo "  [ OK ] $*"; }
bad()  { echo "  [FAIL] $*"; RC=1; }
warn() { echo "  [WARN] $*"; }
info() { echo "  [INFO] $*"; }
RC=0

kernel_ok() {   # true if uname -r >= 6.8.0-101
    local v; v=$(uname -r | grep -oE '^[0-9]+\.[0-9]+\.[0-9]+-[0-9]+' | tr '.-' ' ')
    read -r a b c d <<< "$v"
    (( a > 6 || (a == 6 && b > 8) || (a == 6 && b == 8 && d >= 101) ))
}

stage_check() {
    echo "== 0  check =="
    . /etc/os-release
    [[ "$VERSION_ID" == 24.04 ]] && ok "Ubuntu $VERSION_ID" || bad "Ubuntu $VERSION_ID (need 24.04)"
    kernel_ok && ok "kernel $(uname -r)" || warn "kernel $(uname -r) < 6.8.0-101 (SDK doc minimum; the reference server ran 6.6.0-rc6 fine)"
    if lscpu | grep -qi AuthenticAMD; then
        warn "AMD CPU: BIOS needs SPM=Enabled, CXL Physical Addressing=System address, IOMMU=Enabled"
    else ok "Intel CPU"; fi
    info "XCENA / CXL PCI devices:"; lspci | grep -iE 'cxl|xcena' | sed 's/^/    /'
    lspci | grep -qi nvidia && info "NVIDIA GPU present -> run stage gpu" || info "no NVIDIA GPU -> lanes B2/B3 cannot run here"
    info "NVMe drives (put DATA on the host NVMe, not on the XCENA card):"
    lsblk -d -o NAME,MODEL,SIZE | grep nvme | sed 's/^/    /'
    grep -q memhp_default_state=offline /proc/cmdline && ok "memhp_default_state=offline" \
        || warn "memhp_default_state=offline not on kernel cmdline (needed if devdax reconfigure fails)"
    if curl -sS -m 10 -o /dev/null https://archive.ubuntu.com 2>/tmp/curl.err; then ok "https to archive.ubuntu.com"
    else warn "https to archive.ubuntu.com failed: $(head -c 200 /tmp/curl.err)"
         warn "if it is a hostname mismatch, stage apt switches apt to http (packages are GPG-signed)"; fi
    command -v xcena_cli >/dev/null && ok "xcena_cli installed" || info "xcena_cli not installed yet"
}

stage_apt() {
    echo "== 1  apt =="
    local src=/etc/apt/sources.list.d/ubuntu.sources
    if ! curl -sS -m 10 -o /dev/null https://archive.ubuntu.com 2>/dev/null && grep -q https:// "$src"; then
        sudo cp "$src" "$src.bak" && sudo sed -i 's#https://#http://#g' "$src"
        warn "apt switched to http ($src.bak kept)"
    fi
    sudo apt-get update
    sudo apt-get install -y build-essential cmake ninja-build pkg-config git curl wget jq \
        dkms ndctl daxctl numactl libnuma-dev \
        pciutils nvme-cli fio python3-venv python3-pip python3-dev         || { bad "apt-get install failed (dpkg lock held by unattended-upgrades? wait and rerun)"; return; }
    # Kernel headers: from apt for a stock kernel; a custom kernel must already have
    # /lib/modules/$(uname -r)/build, or mx_dma / nvidia-fs cannot be built.
    if ! sudo apt-get install -y "linux-headers-$(uname -r)" 2>/dev/null; then
        if [[ -e /lib/modules/$(uname -r)/build/Makefile ]]; then info "custom kernel: using $(readlink -f /lib/modules/$(uname -r)/build)"
        else bad "no usable kernel headers: /lib/modules/$(uname -r)/build -> $(readlink -f /lib/modules/$(uname -r)/build) has no Makefile"; fi
    fi
    kernel_ok || warn "kernel $(uname -r) < 6.8.0-101 (SDK's supported minimum); continuing on it"
    [[ $RC == 0 ]] && ok "packages installed"
}

stage_sdk() {
    echo "== 2  sdk =="
    local tb="${1:?usage: setup_new_server.sh sdk <xcena_sdk_vX.Y.Z_ubuntu_24.04.tar.xz>}"
    local sd; sd=$(basename "$tb" | grep -oE '^xcena_sdk_v[0-9.]+[0-9]') || { bad "cannot read SDK version from $tb"; return; }
    info "$sd  sha256: $(sha256sum "$tb" | cut -c1-64)   (v1.4.9: 0fc51233...c73cb)"
    local dir="$HOME/xcena_sdk"; mkdir -p "$dir"
    tar xf "$tb" -C "$dir" || { bad "extract failed"; return; }
    cd "$dir/$sd" || { bad "no $sd in tarball"; return; }
    sudo bash install_dependencies.sh && sudo bash install.sh || { bad "SDK installer failed"; return; }
    cd - >/dev/null
    if ! lsmod | grep -q mx_dma; then
        # The installer must run from the SDK's driver dir (it reads dkms.conf and *.c there);
        # env/scripts/ only keeps a reference copy.
        local drv; drv=$(find "$dir/$sd" -name mxdriver_install.sh | head -1)
        info "mx_dma not loaded -> running ${drv:-<not found>}"
        [[ -n "$drv" ]] && sudo bash "$drv" || bad "mxdriver_install.sh not found in the SDK"
    fi
    command -v xcena_cli >/dev/null && ok "xcena_cli" || bad "xcena_cli missing"
    lsmod | grep -q mx_dma && ok "mx_dma loaded" || bad "mx_dma not loaded"
    systemctl is-active --quiet pxl_resourced && ok "pxl_resourced active" || warn "pxl_resourced not active yet (start after stage dax)"
    sudo xcena_cli fw-info "$DEV"
    info "next: pick a slot that is NOT the Active Slot above, then: bash $0 fw <slot>"
}

stage_compute() {
    echo "== 3  computing mode =="
    # Computing mode is a switch on the running firmware, not a separate build (XCENA, 2026-10-26).
    sudo xcena_cli device-info "$DEV"
    sudo xcena_cli device-info "$DEV" | grep -qE 'Computable *: *Yes' && { ok "already Computable: Yes"; return; }
    sudo xcena_cli imc enable-compute "$DEV" && ok "enable-compute sent" || { bad "imc enable-compute failed"; return; }
    warn "reboot now, then: bash $0 dax ; bash $0 validate"
}

stage_fw() {
    echo "== 3  firmware =="
    local slot="${1:?usage: setup_new_server.sh fw <non-active slot>}"
    local fi; fi=$(sudo xcena_cli fw-info "$DEV")
    local active; active=$(echo "$fi" | grep -i 'Active Slot' | grep -oE '[0-9]+' | head -1)
    [[ "$slot" == "$active" ]] && { bad "slot $slot is the ACTIVE slot - refusing"; return; }
    # Never flash firmware older than the card runs: newer cards need support added in later
    # releases, and an older image can stop the card enumerating on PCIe.
    local cur; cur=$(echo "$fi" | grep -i 'Firmware Revision' | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    if [[ -z "$cur" || "$(printf '%s\n' "$cur" 1.0.10 | sort -V | tail -1)" != 1.0.10 ]]; then
        bad "running firmware ${cur:-unknown} is newer than 1.0.10 (or unreadable) - refusing a downgrade"
        info "a downgrade can stop the card enumerating; ask XCENA first"; return
    fi
    # Same version number exists as a non-Computing bundle, so judge by Computable, not by 1.0.10.
    sudo xcena_cli device-info "$DEV" | grep -qE 'Computable *: *Yes' && { ok "already Computable: Yes, nothing to flash"; return; }
    [[ "${FW_CONFIRM:-}" == yes ]] || { info "FAE confirmed 1.0.10 for this card? rerun with FW_CONFIRM=yes"; return; }
    [[ -f "$FW" ]] || { bad "no $FW (firmware file approved by XCENA)"; return; }
    info "flashing $(basename "$FW")  sha256 $(sha256sum "$FW" | cut -c1-16)..."
    sudo xcena_cli fw-update "$DEV" "$slot" "$FW" || { bad "fw-update failed"; return; }
    sudo xcena_cli fw-info "$DEV"
    warn "now COLD boot: sudo shutdown -h now, power on again (warm reboot is not enough)"
}

stage_dax() {
    echo "== 4  devdax =="
    daxctl list | jq -r '.[] | "\(.chardev)  \(.mode)  \(.size)"' 2>/dev/null | sed 's/^/    /'
    [[ -n "${DAX:-}" ]] || { info "set DAX to the XCENA devices above, e.g. DAX=\"dax0.0\" bash $0 dax"; return; }
    for d in $DAX; do
        sudo daxctl reconfigure-device --mode=devdax --force "$d" && ok "$d devdax" \
            || bad "$d reconfigure failed (add memhp_default_state=offline to GRUB_CMDLINE_LINUX, update-grub, reboot)"
    done
    sudo systemctl restart pxl_resourced && ok "pxl_resourced restarted"
    info "devdax does not survive a reboot: rerun this stage after every boot"
}

stage_validate() {
    echo "== 5  validate =="
    local v; v=$(command -v validate_host.sh || find "$HOME/xcena_sdk" "$HERE/env" -name validate_host.sh 2>/dev/null | head -1)
    if [[ -n "$v" ]]; then bash "$v" 2>&1 | tee "$HERE/results/validate.log"
    else warn "validate_host.sh not found (SDK scripts); skipping the SDK check"; fi
    local di; di=$(sudo xcena_cli device-info "$DEV")
    echo "$di" | grep -qE 'Computable *: *Yes' && ok "Computable: Yes" || bad "Computable != Yes (firmware / cold boot / devdax)"
    echo "$di" | grep -iE 'MU|core' | sed 's/^/    /'
    info "use the MU core count as T for run_lanes.sh (MX1P: 2816 or 3072)"
    sudo xcena_cli im get-smart "$DEV" | grep -E 'ssd_capacity|dram_capacity|prefetch' | sed 's/^/    /'
}

stage_gpu() {
    echo "== 6  gpu + GDS =="
    lspci | grep -qi nvidia || { info "no NVIDIA GPU, skipping"; return; }
    local debs="${1:-}"
    if nvidia-smi >/dev/null 2>&1 && [[ -z "$debs" ]]; then
        info "driver already working ($(nvidia-smi --query-gpu=driver_version --format=csv,noheader)): not touching it"
    elif [[ -n "$debs" ]]; then
        sudo apt-get install -y "$debs"/nvidia-kernel-source-595-server-open_*.deb "$debs"/nvidia-dkms-595-server-open_*.deb
    else
        sudo apt-get install -y nvidia-driver-595-server-open
    fi
    # GDS needs the OPEN kernel module: the proprietary one taints nvidia_fs out of GPL symbols.
    if modinfo nvidia 2>/dev/null | grep -qi 'license: *Dual MIT/GPL'; then
        sudo apt-get install -y nvidia-fs-dkms 2>/dev/null \
            || warn "nvidia-fs-dkms not in apt: add NVIDIA's CUDA repo for ubuntu2404, or build github.com/NVIDIA/gds-nvidia-fs"
    else
        warn "proprietary nvidia module loaded: nvidia_fs cannot load on it, so lane B3 (GDS) is skipped here"
    fi
    sudo modprobe nvidia_fs 2>/dev/null
    lsmod | grep -q nvidia_fs && ok "nvidia_fs loaded" || warn "nvidia_fs not loaded (reboot after a driver install)"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/    /'
    modinfo nvidia 2>/dev/null | grep -qi 'license: *Dual MIT/GPL' && ok "open kernel module" || warn "check that the open nvidia module is the one loaded"
    info "GPU and NVMe host bridges (P2P is slower across bridges):"
    lspci -tv | grep -iE 'nvidia|non-volatile' | sed 's/^/    /'
}

stage_py() {
    echo "== 7  python =="
    python3 -m venv "$VENV" && . "$VENV/bin/activate"
    pip install -U pip numpy
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    if lsmod | grep -q nvidia_fs; then
        pip install kvikio-cu12==26.8.0
        # kvikio drags torch's CUDA libs to 12.9; put the cu124 set back.
        pip uninstall -y cuda-toolkit 2>/dev/null
        pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
    fi
    python - <<'EOF'
import numpy, torch
print("  numpy", numpy.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
try:
    import kvikio, kvikio.defaults as d
    print("  kvikio", kvikio.__version__, "| compat preferred:", d.is_compat_mode_preferred())
except Exception as e:
    print("  kvikio not available:", e)
EOF
    info "activate before run_lanes.sh: source $VENV/bin/activate"
}

mkdir -p "$HERE/results"
case "${1:-}" in
    check) stage_check ;; apt) stage_apt ;; sdk) stage_sdk "${2:-}" ;; fw) stage_fw "${2:-}" ;; compute) stage_compute ;;
    dax) stage_dax ;; validate) stage_validate ;; gpu) stage_gpu "${2:-}" ;; py) stage_py ;;
    *) sed -n 2,17p "$0"; exit 2 ;;
esac
exit $RC
