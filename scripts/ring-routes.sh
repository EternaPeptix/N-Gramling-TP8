#!/usr/bin/env bash
# ring-routes.sh — make every ring node IP-reachable from every other.
# Run once per boot on ALL four nodes (or via systemd unit). Without these,
# cross-subnet fabric traffic dies: the switchless ring gives each link its
# own /24, and Docker sets FORWARD policy DROP.
#
# Ring: 4687 --7275-- 7425 --5bc3-- 4687   (one /24 per link)
set -euo pipefail

# 1. let fabric traffic transit the docker FORWARD policy
sudo iptables -C DOCKER-USER -s 10.100.0.0/16 -d 10.100.0.0/16 -j ACCEPT 2>/dev/null \
  || sudo iptables -I DOCKER-USER -s 10.100.0.0/16 -d 10.100.0.0/16 -j ACCEPT

# 2. routes to the two non-local ring subnets via the port-consistent neighbor
host=$(hostname)
case "$host" in
  spark-4687)
    sudo ip route replace 10.100.225.0/24 via 10.100.224.11 dev rocep1s0f1
    sudo ip route replace 10.100.226.0/24 via 10.100.227.13 dev rocep1s0f0 ;;
  spark-7275)
    sudo ip route replace 10.100.225.0/24 via 10.100.226.12 dev rocep1s0f1
    sudo ip route replace 10.100.224.0/24 via 10.100.227.10 dev rocep1s0f0 ;;
  spark-7425)
    sudo ip route replace 10.100.227.0/24 via 10.100.226.13 dev rocep1s0f1
    sudo ip route replace 10.100.224.0/24 via 10.100.225.11 dev rocep1s0f0 ;;
  spark-5bc3)
    sudo ip route replace 10.100.226.0/24 via 10.100.225.12 dev rocep1s0f0
    sudo ip route replace 10.100.227.0/24 via 10.100.224.10 dev rocep1s0f1 ;;
  *) echo "unknown node $host — edit ring-routes.sh" >&2; exit 1 ;;
esac
echo "ring routes armed on $host"
