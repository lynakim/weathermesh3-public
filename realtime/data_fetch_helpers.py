import sys
import datetime
import time
import os
import builtins


def clean_old(dataset):
    now = datetime.datetime.now(datetime.timezone.utc)
    for f in os.listdir(f"/fast/proc/{dataset}/f000"):
        if f.endswith(".npz") and not f.endswith("tmp.npz"):
            path = f"/fast/proc/{dataset}/f000/{f}"

            if dataset == 'gfs_rt':
                cycle_time = datetime.datetime.fromtimestamp(int(f[:-4]), tz=datetime.timezone.utc)
            else:
                cycle_time = datetime.datetime.strptime(f[:-4], "%Y%m%d%H").replace(tzinfo=datetime.timezone.utc)

            if (now - cycle_time) > datetime.timedelta(days=5):
                os.remove(path)
                print(f"Removed {path} ({(now - cycle_time).days} days old)")


def latest_cycle_time():
    latest_cycle = datetime.datetime.now(tz=datetime.timezone.utc)
    latest_cycle = latest_cycle - datetime.timedelta(microseconds=latest_cycle.microsecond, minutes=latest_cycle.minute, seconds=latest_cycle.second)
    latest_cycle = latest_cycle - datetime.timedelta(hours=latest_cycle.hour % 6)

    return latest_cycle


def parse_time_arg(arg):
    return datetime.datetime.strptime(arg, "%Y%m%d%H").replace(tzinfo=datetime.timezone.utc)


def run_from_argv(download_and_process, dataset, clean_fn=None):
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        timestamp = parse_time_arg(sys.argv[1])

        print(f"Downloading single cycle {timestamp}")
        download_and_process(timestamp)

        if '--clean' in sys.argv:
            if clean_fn is None:
                clean_old(dataset)
            else:
                clean_fn()

        return

    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        if clean_fn is None:
            clean_old(dataset)
        else:
            clean_fn()
        return

    if len(sys.argv) > 2 and (sys.argv[1] == "since" or sys.argv[1] == "between"):
        since = parse_time_arg(sys.argv[2])
        if since.hour % 6 != 0:
            print("WARNING: since time must be a multiple of 6 hours. Coercing to nearest 6-hour boundary.")
            since = since - datetime.timedelta(hours=since.hour % 6)

        end_time = datetime.datetime.now(tz=datetime.timezone.utc)
        if sys.argv[1] == "between":
            end_time = parse_time_arg(sys.argv[3])

        print(f"Downloading data from {since} to {end_time}")

        current_time = since
        while current_time < end_time:
            download_and_process(current_time)
            current_time += datetime.timedelta(hours=6)

        return

    latest_cycle = latest_cycle_time()
    for cycle_num in range(4):
        print(f"{cycle_num + 1}/4")
        download_and_process(latest_cycle - datetime.timedelta(hours=6*cycle_num))
        builtins.print("")
