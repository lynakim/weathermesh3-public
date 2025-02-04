from utils import *


def do_latent_l2(x,outputs):
    if 'latent_l2' not in outputs:
        outputs['latent_l2'] = 0
        outputs['latent_l2_n'] = 0
    outputs['latent_l2'] += torch.mean(x**2)
    outputs['latent_l2_n'] += 1

def make_default_op_map(model):
    # Functions always expect a list of tensors

    op_map = {}
    def encode(x,state,outputs):
        x = model.encoders[0](x[0], state.t0 + 3600*(state.accum_dt - model.encoders[0].mesh.hour_offset))
        do_latent_l2(x,outputs)
        return [x]
    op_map['E'] = encode

    for dh,p in model.processors.items():
        dh = int(dh)
        def proc(x,state,outputs):
            x = p(x[0])
            state.accum_dt += dh
            return [x] 
        op_map[f'P{dh}'] = proc

    def decode(x,state,outputs):
        do_latent_l2(x[0],outputs)
        out = []
        for dec in model.decoders:
            out.append(dec(x[0]))
        return out
    op_map['D'] = decode

    return op_map

def make_targets(todo_dict, xs):
    targets = [SimpleNamespace(
        t0 = xs[-1],
        target_dt=k,
        remaining_steps = v.split(','),
        completed_steps = [],
        tensors = xs[0:-1],
        accum_dt=0,
    ) for k,v in todo_dict.items()]
    return targets

def simple_gen_todo(dts,processor_dts, from_latent=False):
    out = {}
    for dt in dts:
        rem_dt = dt
        todo = "E," if not from_latent else ""
        for pdt in reversed(processor_dts):
            num = rem_dt // pdt
            rem_dt -= pdt*num
            todo+= f"P{pdt},"*num
            if rem_dt == 0:
                break
        assert rem_dt == 0
        todo+="D"
        out[dt] = todo
    return out
 

def process(targets,op_map,callback=None): 
    outputs = {} 

    while targets:
        tnow = targets[0]
        step = tnow.remaining_steps[0]
        completed_steps = tnow.completed_steps.copy()
        x = tnow.tensors

        assert step in op_map.keys(), f"Unkown step {step}"

        x = op_map[step](x,tnow,outputs)

        for todo in targets:
            if todo.remaining_steps[0] == step and todo.completed_steps == completed_steps:
                todo.tensors = x
                todo.remaining_steps = todo.remaining_steps[1:]
                todo.completed_steps.append(step)
                todo.accum_dt = tnow.accum_dt
                if not todo.remaining_steps:
                    if callback is not None:
                        callback(todo.target_dt, x)
                    else:
                        assert todo.target_dt not in outputs, "Key collision in outputs"
                        assert todo.target_dt == todo.accum_dt, f"Target dt {todo.target_dt} != accum_dt {todo.accum_dt}"
                        outputs[todo.target_dt] = x
                    targets.remove(todo)

    if "latent_l2" in outputs:
        outputs["latent_l2"] = outputs["latent_l2"] / outputs["latent_l2_n"]
        del outputs["latent_l2_n"]

    return outputs

