"""Select a conservative cross-NUMA path before starting the full model."""
import json
import os
import pathlib
import re
import subprocess


def selection(topology, mode):
    if mode not in ['auto', 'off', 'on']:
        raise RuntimeError('P2P_MODE must be auto, off or on')
    clean = re.sub(r'\x1b\[[0-9;]*m', '', topology)
    rows = [line.split() for line in clean.splitlines() if re.match(r'^GPU\d+\s', line)]
    if len(rows) != 2 or any(len(row) < 3 for row in rows):
        raise RuntimeError('Expected a topology matrix for the two assigned GPUs')
    # Inspect GPU-to-GPU columns; a remote NIC's SYS path is irrelevant here.
    cross_numa = rows[0][2] == 'SYS' or rows[1][1] == 'SYS'
    fallback = mode == 'off' or (mode == 'auto' and cross_numa)
    return {'mode': mode, 'cross_numa': cross_numa, 'fallback': fallback,
            'NCCL_P2P_DISABLE': '1' if fallback else '0',
            'SGLANG_ENABLE_PCIE_IPC_ALLREDUCE': '0' if fallback else '1'}


if __name__ == '__main__':
    state = pathlib.Path(os.environ.get('STATE_DIR', '/workspace/huihui-glm53-tp2'))
    result = subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise RuntimeError('GPU topology lookup failed before model startup')
    report = selection(result.stdout, os.environ.get('P2P_MODE', 'auto'))
    (state/'communication.json').write_text(json.dumps(report, indent=2)+'\n')
    (state/'communication.env').write_text(
        'export NCCL_P2P_DISABLE='+report['NCCL_P2P_DISABLE']+'\n'+
        'export SGLANG_ENABLE_PCIE_IPC_ALLREDUCE='+report['SGLANG_ENABLE_PCIE_IPC_ALLREDUCE']+'\n')
    print('COMMUNICATION_SELECTED', json.dumps(report), flush=True)
