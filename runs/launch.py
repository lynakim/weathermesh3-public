import sys
sys.path.append('.') # hate to see it, but it's necessary 
import os
import subprocess
import inspect
import socket
import psutil
import argparse
import time
from datetime import datetime
import functools
from utils_lite import *
import threading
from evals.live_validation import run_validation

DEFAULT_PORT = 29500
LOG_PATH = '/fast/runlogs/'


import diskcache as dc
motd_cache = dc.Cache('/tmp/motd_cache')

MOTD = '''

              -*++++++++++++++++++++====-------===+++************#####.               
              .*+++++++++++++++++++++====-------==+++************#####-               
               **++++++++++++++++++++====-------==+++*************%###=               
               **++++++++++++++++++++====-------==+++*************@###+               
               +*++++++++++++++++++++====-------==+++*************@###+               
               =*+++++++++++++++++++++===-------==+++*************%###*               
               #@#++++++++++++++++++++===-------==+++*************####%               
              :##%%@@#++++++++++++++++====-------==++*******##%%@@#@###:              
              ##%%%@@@@@@#++++++++++++====-------==+*%@@@@@%%%%%##@@###-              
              ##%%%@@@@@@@@@@*++++++++====-----*%@@@@@@@@@@%%%%%##%%###+              
             =#%@#++++++++++*#@@+++++++===-#@@%#%%%@@@@@@@@%%%%####%####              
             %%+++++++++++++++++*@+++++=*@*+++++++++++++#@%%%%%##@*%####              
            =*++++++++++++++++++++*@+++%+++++++++++++++++++@%%##@#*%###*.             
            +++++++++++*%*++++++++++%+%++++++++++++++++++++++@##%**#####.             
            *++++++++#%%@@@@+++++++++@++++++++++@@@@*+++++++++@#%***%###:             
            =++++++++*@@@@@%#++++++++*+++++++++#%@@@@@+++++++++@****@###-             
             #++++++++++#@*++++++++++#++++++++++@@@@%@+++++++++%****@###+             
              *+++++++++++++++++++++#++#+++++++++++++++++++++++#****%####             
                #++++++++++++++++++@++++%+++++++++++++++++++++#*****####@             
                 -#%+++++++++++++%++++++==%+++++++++++++++++@*******####%             
                 .*+++++#%%%%*++++++++++===--*#++++++++++%#*********#%###.            
                  **++++++++++++++++++++====-------==+++************#%###-            
                  **++++++++++++++++++++====-------==+++*************@###=            
                  +*++@#*+++++++++++++++====-------==+++*************@###=            
                  -*++*%##@%*+++++++++++====-------==+*##=#**********@###+            
                  -*+++++%@@@@@@%#*+++++====---=+*%@@@@@@%%**********%####            
                  :*++++++++*%@@@@@@@@@@@@@@@@@@@@@@@%#++*************%##%            
                  .**++++++++++++++*##%%%%###**+---===+++*************%###.           
                   **++++++++++++++++++++====-------==+++*************@###-           
                   #*++++++++++++++++++++====-------==+++*************%###*           
                   #*++++++++++++++++++++====-------==+++*************%####           
                   =*++++++++++++++++++++====-------==++++************%###*           
                   :*++++++++++++++++++++====-------==++++************%###*           
                    *++++++++++++++++++++====-------===+++************#%###           
                    *+++++++++++++++++++++====-------==+++*************@###.          

READ THIS MESSAGE BEFORE HITTING ENTER TO CONTINUE
                    
Hello!! This is the message of the day, for when you want to tell peoeple something and you know that zulip is too cluttered for an important PSA.
Please keep the last week of messages in this list, and delete things after that! 

## Friday, January 10th, 2025

Geospatial weights now follow a new shape (re: https://chat.windbornesystems.com/#narrow/stream/201-tech-dlnwp/topic/fixing.20poles/near/6974994)
so that poles are weighted more rather than falling off

FYI, it's really easy to profile stuff now. Just look at train/prof.py. You can profile a run that you have just started with very minimial extra effort.


This message now only shows up when it changes or the first time you run something from a new path on a new node. Please read this message every time and if it gets annoying, poke John.
Understood? Press enter to continue.
'''
@motd_cache.memoize(expire=60*60*24*30)
def message_of_the_day(cwd,motd):
    input(GREEN(motd))

def post_to_zulip(msg):
    os.system('''curl -X POST https://chat.windbornesystems.com/api/v1/messages \
        -u john-bot@chat.windbornesystems.com:9nCdhwXcpxXSxnBaHZHerwGnYnLORmUG \
        -d "type=stream" \
        -d "to=tech-wm-dev" \
        -d "subject=training bots" \
        -d "content=%s" \
    ''' % msg)
    print(msg)

def get_gpu_utilization():
    cmd = "nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader"
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, text=True)
    utilization = result.stdout.strip().replace('%', '')  # Remove the '%' sign if present
    return int(utilization)


def launch(ddp=True,nodes={'localhost':-1},port=DEFAULT_PORT,kill_nvidia=True,clear_cache=True,start_method='spawn',log=True, commit_git=False, zulip=False, ping='Nobody',validate=False):
    def inner(func):
        nonlocal zulip
        global config
        global conf
        #func.launchable = True
        #func.ddp = ddp
        #func.nodes = nodes
        def wrapper():
            config.name = func.__name__.replace('_','-')
            try:
                conf.name = config.name
            except: pass
            from torch.multiprocessing import set_start_method
            set_start_method(start_method)
            try:
                result = func()
                if hasattr(result, '__iter__'):
                    yield from result
                else:
                    yield result
            except KeyboardInterrupt:
                print("KeyboardInterrupt, Bye!")
                sys.exit(0)
        wrapper.__name__ = func.__name__
        wrapper.__qualname__ = func.__qualname__
        wrapper.file = inspect.getfile(func)
        wrapper.launchable = True
        wrapper.ddp = ddp
        wrapper.nodes = nodes
        wrapper.port = port
        wrapper.kill_nvidia = kill_nvidia
        wrapper.clear_cache = clear_cache
        wrapper.log = log
        wrapper.commit_git = commit_git
        if ping != 'Nobody':
            zulip = True
        wrapper.zulip = zulip
        wrapper.ping = ping
        wrapper.validate = validate
        return wrapper
    return inner

def run_training_fn(fn):
    res = fn()
    if hasattr(res, '__iter__'):
        for r in res:
            pass

def get_num_gpus(hostname=None):
    base_cmd = "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"
    cmd = f"ssh {hostname} '{base_cmd}'"
    return int(subprocess.getoutput(cmd))

def free_port(hostname, port=DEFAULT_PORT):
    print("freeing port", port, "on", hostname)
    cmd = ["ssh", hostname, f"lsof -t -i:{port} && (echo '^ {hostname} found this process port {port}, killing it' && kill $(lsof -t -i:{port})) || true"]
    #print(cmd)
    subprocess.run(cmd)

def get_logdir(name):
    return os.path.join(LOG_PATH,name)

def get_logfile(name,hostname):
    return os.path.join(LOG_PATH,name,f'{hostname}.log')

def run(loc):
    message_of_the_day(os.getcwd(),MOTD)
    try: act = int(config.activity)
    except: act = config.activity
    funcs = [f for f in loc if callable(f) and getattr(f,'launchable',False)]
    if type(act) == int:
        func = funcs[act]
        config.activity = func.__name__
    else:
        funcs = [f for f in funcs if f.__name__ == act]
        assert len(funcs) == 1, f"Found {len(funcs)} functions with name {config.activity}"
        func = funcs[0]

    if sys.argv[-2] == 'validate':
        weights = sys.argv[-1]
        assert os.path.exists(weights), f"Could not find weights file {weights}"
        assert func.validate, "This function is not marked as validate"
        model = next(func())
        run_validation(model, weights)
        return
    
    is_torchrun = 'TORCHELASTIC_RUN_ID' in os.environ.keys() 
    if not func.ddp or is_torchrun:
        run_training_fn(func)
        return

    nnodes = len(func.nodes)
    hostname = socket.gethostname()
    assert hostname in func.nodes.keys() or 'localhost' in func.nodes.keys(), f"Hostname {hostname} not in {func.nodes.keys()}"
    port = func.port
    print("nnodes", nnodes)
    mypid = os.getpid()
    zulip_ping = f"echo 'pinging {func.ping}';" if func.zulip else ""
    cmd_str = f'{zulip_ping} export OMP_NUM_THREADS=2 ; export NCCL_IB_HCA=mlx5_0; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;' 

    GOOD_GPUS = {"singing": "0,1,2,4,5", "miramar": "1,2,3,4,5", "bimini": "0,1,2,3,5"}
    GOOD_GPUS["singing"] = "0,1,2,4" 
    #del GOOD_GPUS["singing"]
    #del GOOD_GPUS["bimini"]
    del GOOD_GPUS["miramar"]
    GOOD_GPUS = {"singing": "0,1,2,3,4"}

    if func.kill_nvidia:
        #cmd_str += ' fuser -k /dev/nvidia*;'
        #cmd_str += f'fuser /dev/nvidia* 2>/dev/null | grep -v {mypid} | xargs -r kill;'
        cmd_str += f"fuser /dev/nvidia* 2>/dev/null | tr ' ' '\\n' | grep -v '^{mypid}$' | xargs -r kill;"
        cmd_str += f"rm -f /dev/shm/*;"
        #cmd_str += f"fuser /dev/nvidia* 2>/dev/null | tr ' ' '\\n' | grep -v '^{mypid}$';"
        #subprocess.run(["bash","-c",cmd_str])
    cmd_str += f'torchrun --nnodes={nnodes} --nproc_per_node={{ngpus}} '
    cmd_str += f'--rdzv_id={port} --rdzv_backend=c10d --rdzv_endpoint={hostname}:{port}'
    #cmd_str += f' --master_addr={hostname} --master_port={port}'
    cmd_str += f' {func.file} {config.activity}' 

    log_name = f'{config.activity}.{time.strftime("%b%d_%H%M%S")}'
    print(f'Logging to {get_logdir(log_name)}')
    #config.console_log_path = get_logdir(log_name) # this didn't work as the children dont see this
    os.makedirs(get_logdir(log_name),exist_ok=True)
    if func.commit_git:
        # make a new branch to save repo state
        process = subprocess.Popen(["bash", "-c", 'CURRENT_BRANCH=$(git branch --show-current); git stash -u; TEMP_BRANCH="run-$(date +%Y%m%d%H%M%S)"; git checkout -b $TEMP_BRANCH; git stash apply; git add -A; git commit -m "Temporary commit for run at $(date +%Y%m%d%H%M%S)"; git push origin -u $TEMP_BRANCH; COMMIT_ID=$(git rev-parse HEAD); echo "Run started on commit: $COMMIT_ID, branch: $TEMP_BRANCH"; git checkout $CURRENT_BRANCH; git branch -D $TEMP_BRANCH; git stash pop'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        stdout = stdout.decode()
        print(stdout)
        # log to file
        with open(get_logfile(log_name,hostname), 'a') as f:
            f.write(stdout)
        if stderr:
            stderr = stderr.decode()
            print(stderr, file=sys.stderr)
            with open(get_logfile(log_name,hostname), 'a') as f:
                f.write(stderr)

    for i,(node,ngpus) in enumerate(func.nodes.items()):
        #free_port(node,port=port)
        def logwrap(cmd):
            if func.log:
                cmd = cmd + f" 2>&1 | tee -a {get_logfile(log_name,node)}"
            print(cmd)
            return cmd
        if ngpus == -1:
            ngpus = get_num_gpus(node)
        elif ngpus == 0:
            continue
        devcmd = ""
        nnode = node.split(".")[0]
        if nnode in GOOD_GPUS:
            if nnode == hostname and func.zulip and '0' not in GOOD_GPUS[nnode]:
                assert False, "Notifications will not work unless you are running this on GPU 0"
            devcmd = "export CUDA_VISIBLE_DEVICES="+GOOD_GPUS[nnode]+"; "
        if nnode not in ['halfmoon'] and func.clear_cache:
            ohp = False
            for p in psutil.process_iter():
                if 'drop_caches' in ' '.join(p.cmdline()):
                    ohp = True
                    break
            if not ohp:
                devcmd += "echo clearing caches; echo sswa | sudo -S bash -c 'echo 1 > /proc/sys/vm/drop_caches'; echo page cache cleared; "
            else:
                print("woah buddy i see you're spamming new runs, i'll let you start the run without clearing cache")
        lcmd = logwrap(devcmd + cmd_str.format(ngpus=ngpus))
        print(RED(f'Launching on host {node}'))
        if node == hostname:
            print(lcmd)
            subprocess.Popen(["bash","-c",lcmd])
            continue

        lcmd = f'cd {os.getcwd()} && {lcmd}'
        lcmd = f'cd {os.getcwd()} && source /home/windborne/.bashrc &&{lcmd}'
        cmd = ['ssh', node, f'bash -l -c "{lcmd}"']
        subprocess.Popen(cmd)

    zulip_string = f'{config.activity} on {socket.gethostname()}. Nodes: {func.nodes}. Pinging {func.ping}'
    if func.zulip:
        print("Notifying Zulip")
        post_to_zulip(f"Started {zulip_string}".replace("@","@_"))

    print(func.nodes[hostname])
    print(RED("Mother process launched all children"))
    time.sleep(10)
    threading.Thread(target=health_thread,args=(func.zulip,zulip_string)).start()
    while True:
        children = psutil.Process().children(recursive=True)
        if len(children) < 2:
            print(RED(f"[Mother] Only {len(children)} left, exiting"))
            if func.zulip:
                post_to_zulip(f"Looks like this run, mother lost her children :cry:. {zulip_string}")
            sys.exit(0)
        status = [c.status() for c in children]
        # print all differnt statuses and the counts
        sts = set(status)
        from collections import Counter
        c = Counter(status)

        st = f"[Mother {datetime.now().strftime('%b%d %H:%M')}] {len(children)} children. "
        for s in sts:
            st += f"{s}:{c[s]} "
        print(RED(st))
        time.sleep(60)

from collections import deque
def health_thread(zulip,run_string):
    posted_to_zulip = False
    time.sleep(20*60)
    buf = deque(maxlen=6*10) # ~10 minutes
    while True:
        buf.append(get_gpu_utilization())
        if len(buf) == buf.maxlen:
            max_util = max(buf)
            if max_util < 10:
                if zulip and not posted_to_zulip:
                    print(RED(f"[Health] GPU utilization is low: {buf}"))
                    post_to_zulip(f"Looks like this run failed, GPU's are sad :pain:. {run_string}")
                    posted_to_zulip = True
            if posted_to_zulip and max_util > 80:
                print(RED(f"[Health] GPU utilization is back to normal: {buf}"))
                post_to_zulip(f":dancing-eagle: we're back in business {run_string}")
                posted_to_zulip = False
        time.sleep(10)


def parse_args():
    parser = argparse.ArgumentParser(description='Ohp')
    parser.add_argument("activity", type=str, nargs='?',default="None", help="activity to tun")
    #parser.add_argument("gpus", type=int, nargs='?',default=1, help="number of gpus")
    args, unk = parser.parse_known_args()
    config = WeatherTrainerConfig(**args.__dict__)
    config.start_time = time.time()
    return config


config = parse_args()

try:
    from hres.hres_utils import *
    conf = default_config
except:
    pass
