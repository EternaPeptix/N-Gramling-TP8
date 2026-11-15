#!/usr/bin/env bash
# ring-routes.sh — make every ring node IP-reachable from every other.
# Run once per boot on ALL four nodes (or via systemd unit). Without these,
# cross-subnet fabric traffic dies: the switchless ring gives each link its
# own /24, and Docker sets FORWARD policy DROP.
#
# The fabric netdev names flap (enp1s0* <-> rocep1s0* on driver reload), so
# every rule is resolved from the 10.100.x addresses, never a hardcoded name.
set -euo pipefail

# 1. let fabric traffic transit the docker FORWARD policy
sudo iptables -C DOCKER-USER -s 10.100.0.0/16 -d 10.100.0.0/16 -j ACCEPT 2>/dev/null \
  || sudo iptables -I DOCKER-USER -s 10.100.0.0/16 -d 10.100.0.0/16 -j ACCEPT

# 2. routes to the two non-local ring subnets via the port-consistent neighbor
#    (interface resolved by the address it holds, name-agnostic)
if0=$(ip -br addr show | awk '$3 ~ /^10\.100\./{print $1; exit}')
if1=$(ip -br addr show | awk '$3 ~ /^10\.100\./{print $1}' | sed -n 2p)
[ -n "$if0" ] && [ -n "$if1" ] || { echo "expected two 10.100.x fabric interfaces" >&2; exit 1; }
ip0=$(ip -br addr show dev "$if0" | awk '{print $3}' | cut -d/ -f1)
ip1=$(ip -br addr show dev "$if1" | awk '{print $3}' | cut -d/ -f1)

host=$(hostname)
case "$host" in
  spark-4687)
    sudo ip route replace 10.100.225.0/24 via 10.100.224.11 dev "$if1"
    sudo ip route replace 10.100.226.0/24 via 10.100.227.13 dev "$if0" ;;
  spark-7275)
    sudo ip route replace 10.100.225.0/24 via 10.100.226.12 dev "$if1"
    sudo ip route replace 10.100.224.0/24 via 10.100.227.10 dev "$if0" ;;
  spark-7425)
    sudo ip route replace 10.100.227.0/24 via 10.100.226.13 dev "$if1"
    sudo ip route replace 10.100.224.0/24 via 10.100.225.11 dev "$if0" ;;
  spark-5bc3)
    sudo ip route replace 10.100.226.0/24 via 10.100.225.12 dev "$if0"
    sudo ip route replace 10.100.227.0/24 via 10.100.224.10 dev "$if1" ;;
  *) echo "unknown node $host — edit ring-routes.sh" >&2; exit 1 ;;
esac
echo "ring routes armed on $host ($if0=$ip0 $if1=$ip1)"
