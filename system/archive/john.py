import psutil
import time

while True:
    old_value = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv

    time.sleep(1)

    new_value = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv

    print(f"Network usage: {(new_value - old_value)/1000_000} MB/sec")