import subprocess
import datetime
import os
import sys
ONPREM_NODES = ['stinson','halfmoon','barceloneta','singing','miramar','bimini','baga','muir','glass', 'gold']
CLOUD_NODES = [] #'nimbus']#, 'cirrus']
NODES = ONPREM_NODES + CLOUD_NODES

def ls_gpu():
    global NODES
    for node in NODES:
        cmd = ['ssh', '-o','ConnectTimeout=1',node,'echo $(hostname) && nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader']
        subprocess.run(cmd)

def ls_runs():
    for node in NODES:
        print(f'--- {node} ---')
        cmd = ['ssh', node, 'ps -aux --no-headers']
        r = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output = r.stdout.read().decode('utf-8')
        for line in output.splitlines():
            if 'torchrun' in line:
                if '/home/windborne/.local/bin/torchrun' in line:
                    continue
                #print(line)
                date, nodes, name, ssh, host , who_ping, num_gpus =  parse_line(line)
                try:
                    cloud_ssh_str = f'ssh {node} ' if node in CLOUD_NODES else ''
                    if node in CLOUD_NODES:
                        runfile = subprocess.run(f"{cloud_ssh_str} cat {name.split()[0]}", shell=True, capture_output=True, text=True).stdout
                    else:
                        with open(name.split()[0]) as f:
                            runfile = f.read()
                    runfn = runfile[runfile.index(name.split()[1]+"("):]
                    runfn = runfn[:runfn.index(".run()")] 
                    sched = eval(re.findall('lr_sched.cosine_period\s*=\s*(.*)', runfn)[0])
                    try:
                        prefix = eval(re.findall('config.prefix\s*=\s*(.*)', runfn)[0])
                    except:
                        prefix = ""
                    
                    filecmd = subprocess.run(f"{cloud_ssh_str}ls -ltr /huge/deep/runs{prefix}/`ls -tr /huge/deep/runs{prefix} | grep %s_ | tail -n1` | grep model | tail -n2" % (name.split()[-1].replace("_", "-")), shell=True, capture_output=True, text=True).stdout
                    last_fn = filecmd.split('\n')[-2]
                    step_last = int(re.findall('step(\d+)',last_fn)[0])
                    
                    stepstr = " | %d/%d (%.1f%%)" % (step_last, sched, step_last/sched*100)
                except:
                    stepstr = ""

                try:    
                        prev_fn = filecmd.split('\n')[-3]
                        t_last = datetime.datetime.strptime(' '.join([x for x in last_fn.split(' ') if x != ''][5:8]), '%b %d %H:%M').replace(year=2024)
                        t_prev = datetime.datetime.strptime(' '.join([x for x in prev_fn.split(' ') if x != ''][5:8]), '%b %d %H:%M').replace(year=2024)
                        step_prev = int(re.findall('step(\d+)',prev_fn)[0])
                        remain = (sched - step_last) * (t_last - t_prev).total_seconds() / (step_last - step_prev)
                        # convert seconds to days hours minutes
                        remain = datetime.timedelta(seconds=remain)
                        remainstr = " | %dd%02dh remaining" % (remain.days, remain.seconds//3600)
                except:
                    remainstr = ""    
                
                sshstr = 'local'
                if ssh:
                    sshstr = f'-> {host}'
                print(f'{date} | {name} | {nodes} nodes | {num_gpus} gpus{stepstr}{remainstr} | {sshstr} | pinging {who_ping}')


import re

def parse_line(line):
    # Extract date
    date_pattern = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\d{2}\b'
    date_match = re.search(date_pattern, line)
    date = date_match.group(0) if date_match else None

    # Extract number of nodes
    nodes_pattern = r'--nnodes=(\d+)'
    nodes_match = re.search(nodes_pattern, line)
    nodes = int(nodes_match.group(1)) if nodes_match else None
    
    last_slash = [i for i,word in enumerate(line.split(' ')) if 'runs/' in word or 'hres/run' in word or 'neopointy/run' in word][-1]
    last_arg = " ".join(line.split(' ')[last_slash:last_slash+2])
    #if "| tee" in line:
    #    last_arg = " ".join(line.split("| tee")[0].split(' ')[-3:])
    #else:
    #    last_arg = " ".join(line.split(' ')[-3:])

    # Check for SSH and extract hostname if present
    ssh_pattern = r'\bssh\s+(\S+)'
    ssh_match = re.search(ssh_pattern, line)
    ssh = ssh_match is not None
    hostname = ssh_match.group(1) if ssh else None

    who_ping = "nobody"
    if "pinging" in line:
        who_ping = line.split("pinging")[1].split(";")[0]

    num_gpus = int(line.split('--nproc_per_node=')[1].split()[0])
    num_gpus = num_gpus * nodes
    return date, nodes, last_arg, ssh, hostname, who_ping, num_gpus



import subprocess
import threading
import re
import numpy as np

def parse_nvidia_smi_output(output):
    """Parse the output of nvidia-smi and return the average metrics."""
    lines = output.strip().split('\n')
    gpu_utils, mem_usages, power_draws = [], [], []

    for line in lines:
        if '%' in line:
            # Parse utilization, memory, and power
            parts = re.findall(r'(\d+) %, (\d+) MiB, (\d+.\d+) W', line)
            if parts:
                util, mem, power = map(float, parts[0])
                gpu_utils.append(util)
                mem_usages.append(mem)
                power_draws.append(power)

    # Calculate averages
    avg_gpu_util = np.mean(gpu_utils) if gpu_utils else 0
    avg_mem_usage = np.mean(mem_usages) if mem_usages else 0
    avg_power_draw = np.mean(power_draws) if power_draws else 0

    return avg_gpu_util, avg_mem_usage, avg_power_draw

def run_nvidia_smi(node, duration, interval, result_dict):
    """Run nvidia-smi command on a remote node and parse the output."""
    cmd = f"ssh {node} 'echo $(hostname); for i in {{1..{int(duration/interval)}}}; do nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader; sleep {interval}; done'"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, text=True)
    output = process.communicate()[0]

    # Parse the output
    result_dict[node] = parse_nvidia_smi_output(output)

def ls_gpu_avg(duration=10, interval=0.5):
    nodes = NODES
    threads = []
    results = {}

    for node in nodes:
        thread = threading.Thread(target=run_nvidia_smi, args=(node, duration, interval, results))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    for k,v in results.items():
        print(f'{k:15}: {v[0]:10.1f}% {v[1]:10.1f}MiB {v[2]:10.1f}W')


def ls_screens():
    global NODES
    for node in NODES:
        cmd = ['ssh', node,'screen -ls']
        r = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output = r.stdout.read().decode('utf-8').splitlines()[1:-1]
        for line in output:
            print(f'{node:15} {line[1:]}')
        print('---')
        #print(output)


if __name__ == '__main__':
    #ls_screens()
    ls_runs()
