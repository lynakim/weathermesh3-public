export OMP_NUM_THREADS=8 

#torchrun --nnodes=2 \
#    --nproc_per_node=2 \
#    --rdzv_id=100 \
#    --rdzv_backend=c10d \
#    --rdzv_endpoint=halfmoon:29400 \
#    runs/Oct7.py Oct17-DDPtest

torchrun --nproc_per_node=2 runs/Oct7.py Oct17-DDPtest
