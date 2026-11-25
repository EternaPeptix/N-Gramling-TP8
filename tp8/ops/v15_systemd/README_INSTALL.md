# Wave-V15 Systemd Installation & Operations Guide (`v15_systemd/`)

Install these units on the **head node (`spark-one`)** only. The supervisor orchestrates all 8 DGX Spark nodes (`spark-one`..`spark-eight`) over passwordless SSH (using `ProxyJump` through `spark-one` for fabric-only nodes `spark-five`/`spark-six`).

## 1. Stage Scripts & Generate `/rdma/SHA256SUMS` on All 8 Nodes

```bash
export V15_USER="$(whoami)"
export V15_INSTALL_DIR="/home/${V15_USER}/inkling-tp8"

# Copy launcher + watchdog + UMA guard + supervisor to V15_INSTALL_DIR on spark-one
# and distribute inkling-tp8-v15.sh + /rdma/SHA256SUMS to all 8 nodes:
chmod +x "${V15_INSTALL_DIR}/inkling-tp8-v15.sh" \
         "${V15_INSTALL_DIR}/v15_preboot_uma_guard.sh" \
         "${V15_INSTALL_DIR}/v15_cluster_supervisor.sh" \
         "${V15_INSTALL_DIR}/v15_health_watchdog.py"

# Lock /rdma/*.so SHA-256 manifest before launching:
bash "${V15_INSTALL_DIR}/inkling-tp8-v15.sh" --generate-rdma-manifest "${V15_INSTALL_DIR}/rdma"
```

## 2. Configure `/etc/inkling-v15/supervisor.env` (Zero Hardcoded Secrets)

```bash
sudo mkdir -p /etc/inkling-v15 /var/log/inkling-v15 /var/lib/inkling-v15
sudo cp v15_systemd/supervisor.env.example /etc/inkling-v15/supervisor.env
sudo sed -i "s|REPLACE_WITH_CLUSTER_USER|${V15_USER}|g" /etc/inkling-v15/supervisor.env
sudo chown "root:${V15_USER}" /etc/inkling-v15/supervisor.env
sudo chmod 0640 /etc/inkling-v15/supervisor.env
# Edit /etc/inkling-v15/supervisor.env to confirm V15_PRIMARY_ENV & V15_FALLBACK_ENV paths.
```

## 3. Optional Minimal Passwordless Sudoers (All 8 Nodes, For UMA Defrag)

Install `/etc/sudoers.d/inkling-v15-uma` (`chmod 0440`, validated with `visudo -c`) on all 8 nodes so `v15_preboot_uma_guard.sh` can run non-destructive cache drop / memory compaction / swap drain (and optional `--auto-reboot-defrag` when explicitly invoked):

```sudoers
REPLACE_WITH_CLUSTER_USER ALL=(root) NOPASSWD: /usr/sbin/sysctl -w vm.drop_caches=3, /usr/sbin/sysctl -w vm.compact_memory=1, /usr/sbin/swapoff -a, /usr/sbin/swapon -a, /usr/sbin/reboot
```

## 4. Render & Install Systemd Units on `spark-one`

```bash
for unit in inkling-tp8-v15-supervisor.service inkling-tp8-v15-watchdog.service inkling-tp8-v15-watchdog.timer; do
  sed -e "s|@V15_USER@|${V15_USER}|g" \
      -e "s|@V15_INSTALL_DIR@|${V15_INSTALL_DIR}|g" \
      "v15_systemd/${unit}" | sudo tee "/etc/systemd/system/${unit}" >/dev/null
done

sudo systemctl daemon-reload
sudo systemd-analyze verify \
  /etc/systemd/system/inkling-tp8-v15-supervisor.service \
  /etc/systemd/system/inkling-tp8-v15-watchdog.service \
  /etc/systemd/system/inkling-tp8-v15-watchdog.timer
```

## 5. Enable, Inspect & Operate

```bash
# Enable & start 24/7 supervisor + periodic watchdog timer:
sudo systemctl enable --now inkling-tp8-v15-supervisor.service
sudo systemctl enable --now inkling-tp8-v15-watchdog.timer

# Live status & logs:
sudo systemctl status inkling-tp8-v15-supervisor.service inkling-tp8-v15-watchdog.timer
journalctl -u inkling-tp8-v15-supervisor.service -u inkling-tp8-v15-watchdog.service -f

# Manual 8-node dashboard & crash bundle:
bash "${V15_INSTALL_DIR}/v15_cluster_supervisor.sh" status
bash "${V15_INSTALL_DIR}/v15_cluster_supervisor.sh" bundle manual_check

# Reset circuit breaker & V12 fallback lock after repairing a hardware/link fault:
bash "${V15_INSTALL_DIR}/v15_cluster_supervisor.sh" reset-breaker
```

## 6. Clean Disable / Rollback

```bash
sudo systemctl disable --now inkling-tp8-v15-watchdog.timer inkling-tp8-v15-supervisor.service
bash "${V15_INSTALL_DIR}/v15_cluster_supervisor.sh" stop
# Launch live_champion_v12.env directly if needed:
bash "${V15_INSTALL_DIR}/v15_cluster_supervisor.sh" start /path/to/live_champion_v12.env
```
