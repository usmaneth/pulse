#!/usr/bin/env python3
"""Check the local node or SSH nodes against a lease process allowlist."""
import json
import subprocess
import sys

PROBE = r'''
import json, shutil, subprocess
mem = dict(line.split(':',1) for line in open('/proc/meminfo'))
p = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
pids = [int(x.strip()) for x in p.stdout.splitlines() if x.strip()]
print(json.dumps({'mem_available_gib': int(mem['MemAvailable'].split()[0])/1048576, 'disk_free_gib': shutil.disk_usage('/home').free/2**30, 'gpu_pids':pids}))
'''


def main():
    body = json.load(sys.stdin)
    nodes = {}
    exclusive = True
    for node in body["nodes"]:
        argv = ["python3", "-c", PROBE] if node == "localhost" else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node, "python3 -c " + __import__("shlex").quote(PROBE)]
        proc = subprocess.run(argv, text=True, capture_output=True, timeout=30, check=True)
        nodes[node] = json.loads(proc.stdout)
        expected = set(body["lease"]["allowed_gpu_pids"][node])
        observed = set(nodes[node]["gpu_pids"])
        exclusive &= bool(expected) and observed == expected
    print(json.dumps({"exclusive": exclusive, "nodes": nodes}))


if __name__ == "__main__":
    main()
