mkdir -p tools/nsight/
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --cudabacktrace=true -x true -o tools/nsight/$1 python3 evals/run_compute.py