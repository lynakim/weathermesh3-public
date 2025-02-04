#!/bin/bash

while true; do
/fast/parsudocmd.sh "sudo sh -c 'echo 1 > /proc/sys/vm/compact_memory'";
/fast/parsudocmd.sh "sudo sh -c 'echo 1 > /proc/sys/vm/drop_caches'";
sleep 200;
done
