import torch
import os 


# How to profile a training run
#
# Look at the code at the bottom of this file. It's very simple, follow the instructions there.
# 
# To use nvidia nsight (I use nsight and nsys interchangably, I forget which each term referse to exactly)
# you simple run nsys.sh. That script runs this python function wrapped in nsys
# 
# For the torch profiling outputs, look at the comments in that function for how to view them.

# Useful links:
# https://pytorch.org/docs/stable/torch_cuda_memory.html
# https://pytorch.org/memory_viz
#



def profile_torch(trainer):
    # Pytorch offers a number of profiling tools and I'm only using a fraction of them here. Feel free to add more, but keep the code clean
    # and make sure things can be enabled and disabled easily. 

    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1),
            #on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )
    torch.cuda.memory._record_memory_history()
    trainer.setup()
    prof.start()
    for i in range(3):
        trainer.train()
        prof.step()
    prof.stop()
    prof.export_memory_timeline('ignored/perf/memory_timeline.html')
    torch.cuda.cudart().cudaProfilerStop()

    # To view the memory snapshot pickle, download it and then upload it to https://pytorch.org/memory_viz to view.  
    torch.cuda.memory._dump_snapshot("ignored/perf/memory_snapshot.pickle") 

def profile_nsys(trainer):

    trainer.setup()
    for i in range(3):
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("iteration{}".format(i))
        trainer.train()
        torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == '__main__':

    # in your run function, return the trainer at the very end rather than calling .run()
    # then, import it here as run_fn to profile it. 

    #from runs.jd import Jan7_daforecast_js as run_fn
    from runs.jd import Feb1_john_pointy_regioned as run_fn
    trainer = next(run_fn()) 

    if 'NSYS_TARGET_STD_REDIRECT_ACTION' in os.environ:
        profile_nsys(trainer)
    else:
        profile_torch(trainer)
