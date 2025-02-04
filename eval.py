from matplotlib.pyplot import pie
import torch
import torch.nn.functional
import torch.distributed as dist
from utils import * 

def eval_rms_train(preds, actuals, norm_matrix, weight, keys=None, by_level=False, stdout=True, ddp_reduce=True, mesh=None):
    # this is the vesion called by train.py
    # it's slightly different as it has deal with some stuff being on GPU
    with torch.no_grad():
        B,N1,N2,D = preds.shape[:4]
        if keys is None:
            keys = ['var'+str(i) for i in range(D)]
        assert len(preds.shape) <= 5, f"preds must have shape (B,N1,N2,D) or (B,N1,N2,D,O)"
        assert preds.shape == actuals.shape, f"{preds.shape} != {actuals.shape}"
        assert D == norm_matrix.shape[0], f"preds and norm must have same number of variables. preds: {preds.shape}, norm: {norm_matrix.shape}"
        assert preds.shape[1:3] == weight.shape, f"preds and weights must have same spacial dims. preds: {preds.shape}, weight: {weight.shape}"

        error = preds - actuals

        O = 1 if len(preds.shape) == 4 else preds.shape[4]
        levs = mesh.levels

        ws = torch.sum(weight)*B*O
        rms_ = []
        rms_dict = {}
        extra = {}
        for i,k in enumerate(keys):
            f = 1e6 if k == '133_q' else 1 # use mg/kg instead of kg/kg for Q
            e = error[:,:,:,i] * norm_matrix[i]
            if k in vars_with_nans:
                # Create a mask for non-NaN values
                mask = ~torch.isnan(e)
                # Replace NaN with 0 in e for the einsum operation
                e = torch.nan_to_num(e, nan=0.0)
                # Compute the sum of weights for non-NaN values
                ws = torch.einsum('bnm...,nm->',mask.float(), weight)
                if ws == 0: ws = 1
            #print("uhhhh", e.dtype, "sq", torch.square(e).dtype, "weight", weight.dtype, "preds", preds.dtype, "error", error.dtype)
            msall = torch.einsum('bnm...,nm->',torch.square(e),weight)/ws * f**2
            if ddp_reduce:
                dist.all_reduce(msall, op=dist.ReduceOp.SUM)
                rmsall = torch.sqrt(msall / dist.get_world_size())
            else:
                rmsall = torch.sqrt(msall)
            rmsall = rmsall.to('cpu').numpy()
            if stdout: print(f"{k}: {rmsall:.2f}")
            if by_level and O > 1:
                #print(e.shape,weight.shape)
                mslev = torch.einsum('bnml,nm->l',torch.square(e),weight)/ws*O * f**2
                if ddp_reduce:
                    dist.all_reduce(mslev, op=dist.ReduceOp.SUM)
                    rmslev = torch.sqrt(mslev / dist.get_world_size())
                else:
                    rmslev = torch.sqrt(mslev)
                rmslev = rmslev.to('cpu').numpy()
                for j in range(O):
                    extra[k+"_"+str(levs[j])] = float(rmslev[j])
                    if stdout: print(f"  {k} {levs[j]}: {rmslev[j]:.2f}")
            rms_.append(rmsall)
            rms_dict[k] = float(rmsall)
        if by_level: rms_dict.update(extra)
        return rms_dict

def eval_rms_point(preds, actuals, weights, means, stds, ddp_reduce=True):
    """
    preds: shape (B, 8) for the 8 output variables
    actuals: shape (B, 8) for the 8 output variables
    weights: shape (B,) for the weights
    means: shape (8,) for the means of the variables
    stds: shape (8,) for the stds of the variables
    """
    with torch.no_grad():
        B, D = preds.shape
        assert D == actuals.shape[1], f"{D} != {actuals.shape[1]}"
        assert B == actuals.shape[0], f"{B} != {actuals.shape[0]}"
        assert B == weights.shape[0], f"{B} != {weights.shape[0]}"
        assert D == means.shape[0], f"{D} != {means.shape[0]}"
        assert D == stds.shape[0], f"{D} != {stds.shape[0]}"

        if means.device != preds.device:
            means = means.to(preds.device)
            stds = stds.to(preds.device)
        error = preds - actuals
        error = error * stds

        rms_ = []
        for i in range(D):
            e = error[:, i]
            mask = ~torch.isnan(actuals[:, i])
            e = e[mask]
            w = weights[mask]
            ws = torch.sum(w)
            if ws == 0: ws = 1
            msall = torch.einsum('b,b->', torch.square(e), w) / ws
            if ddp_reduce:
                dist.all_reduce(msall, op=dist.ReduceOp.SUM)
                rmsall = torch.sqrt(msall / dist.get_world_size())
            else:
                rmsall = torch.sqrt(msall)
            rmsall = rmsall.to('cpu').numpy()
            rms_.append(rmsall)
        return rms_

def meingott(e,weight):
    if weight is None:
        weight = torch.ones_like(e)
    ll = 'abcdefghijklmnopqrstuvwxyz'
    a = [np.where(np.array(e.shape) == weight.shape[i])[0][0] for i in range(len(weight.shape))]
    nota = [i for i in range(len(e.shape)) if i not in a]
    einstring = f'{ll[:len(e.shape)]},{"".join([ll[x] for x in a])}->{"".join([ll[x] for x in nota])}'
    weight_sum = torch.sum(weight)
    #weight_sum *= np.prod([e.shape[x] for x in nota])
    #print(einstring)
    return weight, weight_sum, einstring

def _rmse(e,weight=None):
    if type(e) != torch.Tensor:
        e = torch.tensor(e)
    weight, weight_sum, einstring = meingott(e,weight)
    return torch.sqrt(torch.einsum(einstring,torch.square(e),weight)/weight_sum)

def _mse(e,weight=None):
    weight, weight_sum, einstring = meingott(e,weight)
    return torch.einsum(einstring,torch.square(e),weight)/weight_sum

def _bias(e,weight=None):
    weight, weight_sum, einstring = meingott(e,weight)
    return torch.einsum(einstring,e,weight)/weight_sum

def _mae(e,weight=None):
    weight, weight_sum, einstring = meingott(e,weight)
    return torch.einsum(einstring,torch.abs(e),weight)/weight_sum

all_metric_fns = {
    'rmse': _rmse,
    'mse': _mse,
    'bias': _bias,
    'mae': _mae,
}

def eval_metric(preds,actuals,weights,mesh,keys=None,by_level=False, stdout=True, bbox=None, metric_fn=_rmse, resolution=0.25, bylat=False):
    #stdout = True
    # this version is used outside of training
    with torch.no_grad():
        if len(preds.shape) == 2: preds = preds.unsqueeze(2).unsqueeze(0)
        if len(actuals.shape) == 2: actuals = actuals.unsqueeze(2).unsqueeze(0)
        B,N1,N2,D = preds.shape[:4]
        weights = torch.tensor(weights, dtype=torch.float32).to(preds.device)  
        if keys is None:
            keys = ['var'+str(i) for i in range(D)]
        assert len(preds.shape) <= 5, f"preds must have shape (B,N1,N2,D) or (B,N1,N2,D,O)"
        assert preds.shape == actuals.shape, f"{preds.shape} != {actuals.shape}"
        if resolution != 0.25:
            ds = int(preds.shape[1]/weights.shape[1]) # downsample factor
            preds = preds[:, ::ds,::ds, :]
            actuals = actuals[:, ::ds,::ds, :]
        assert preds.shape[:3] == weights.shape[:3], f"preds and weights must have same spatial dims. preds: {preds.shape}, weight: {weights.shape}"
        error = preds - actuals

        O = 1 if len(preds.shape) == 4 else preds.shape[4]
        levs = mesh.levels
        #levs = levels_medium
        #if O == len(levels_ecm2): levs = levels_ecm2
        #if O == len(levels_tiny): levs = levels_tiny

        if bbox is not None:
            error = select_bbox(error,mesh,bbox)
            weight = select_bbox(weight,mesh,bbox)

        #ws = torch.sum(weight)*B*O
        rms_ = []
        extra = {}
        byl = {}
        
        if len(error.shape) == 4:
            error = error.unsqueeze(-1)
        for i,k in enumerate(keys):
            f = 1e6 if k == '133_q' else 1
            e = error[:,:,:,i]
            #### AAAAAA make this not be dumb 
            rmsall = metric_fn(metric_fn(e,weights)).to('cpu').numpy() * f
            if stdout: print(f"{k}: {rmsall:.2f}")
            if by_level and O > 1:
                ##### AAAAAAAA I bet this is wrong cause of O but idk how 
                assert B == 1, "need to modify this"
                rmslev = metric_fn(e,weights).to('cpu').numpy() * f
                if bylat:
                    ohp = None
                    if metric_fn.__name__ == '_rmse':
                        ohp = torch.sqrt(torch.mean(torch.square(e), axis=(0,2)))
                    elif metric_fn.__name__ == '_bias':
                        ohp = torch.mean(e, axis=(0,2))
                    else:
                        assert ValueError("Bylat not implemented for this")
                    assert ohp is not None
                    byl[k] = ohp.to('cpu').numpy()
                for j in range(O):
                    extra[k+"_"+str(levs[j])] = float(rmslev[j])
                    if stdout: print(f"  {k} {levs[j]}: {rmslev[j]:.2f}")
            rms_.append(rmsall)
            extra[k] = float(rmsall)
        if bylat:
            return rms_, extra, byl
        return rms_, extra


def eval_rms(preds,actuals,mesh,keys=None,by_level=False, stdout=True, bbox=None):
    # this version is used outside of training

    with torch.no_grad():
        B,N1,N2,D = preds.shape[:4]
        weights = torch.Tensor(weights).to(preds.device)  
        if keys is None:
            keys = ['var'+str(i) for i in range(D)]
        assert len(preds.shape) <= 5, f"preds must have shape (B,N1,N2,D) or (B,N1,N2,D,O)"
        assert preds.shape == actuals.shape, f"{preds.shape} != {actuals.shape}"
        assert preds.shape[1:3] == weights.shape, f"preds and weights must have same spacial dims. preds: {preds.shape}, weight: {weights.shape}"
        
        error = preds - actuals

        O = 1 if len(preds.shape) == 4 else preds.shape[4]
        levs = mesh.levels
        #levs = levels_medium
        #if O == len(levels_ecm2): levs = levels_ecm2
        #if O == len(levels_tiny): levs = levels_tiny

        if bbox is not None:
            error = select_bbox(error,mesh,bbox)
            weights = select_bbox(weights,mesh,bbox)

        ws = torch.sum(weights)*B*O
        rms_ = []
        extra = {}
        for i,k in enumerate(keys):
            f = 1e6 if k == '133_q' else 1
            e = error[:,:,:,i] 
            rmsall = torch.sqrt(torch.einsum('bnm...,nm->',torch.square(e),weights)/ws).to('cpu').numpy() * f
            if stdout: print(f"{k}: {rmsall:.2f}")
            if by_level and O > 1:
                #print(e.shape,weight.shape)
                rmslev = torch.sqrt(torch.einsum('bnml,nm->l',torch.square(e),weights)/ws*O).to('cpu').numpy() * f
                for j in range(O):
                    extra[k+"_"+str(levs[j])] = float(rmslev[j])
                    if stdout: print(f"  {k} {levs[j]}: {rmslev[j]:.2f}")
            rms_.append(rmsall)
            extra[k] = float(rmsall)
        if by_level: return rms_, extra
        return rms_, extra




def compute_errors(y,yt,weights,mesh,doprint=0,bbox=None,metric_fn=_rmse,only_sfc=False, resolution=0.25, bylat=False):
    pressure_vars = mesh.pressure_vars
    sfc_vars = mesh.sfc_vars
    with torch.no_grad():

        B,N1,N2,D = y.shape 
        nL = mesh.n_levels
        nP = mesh.n_pr_vars
        nPL = mesh.n_pr
        nS = mesh.n_sfc_vars

        if not only_sfc:
            eval_shape = (B,N1,N2,nP,nL)
            to_eval = lambda z : z[...,:nPL].view(eval_shape)
            pred = to_eval(y)
            actual = to_eval(yt)
            out = eval_metric(pred,actual,weights,mesh,keys=pressure_vars,by_level=True,stdout=doprint,bbox=bbox,metric_fn=metric_fn, resolution=resolution, bylat=bylat)
            if bylat:
                _, rms, byl = out
            else:
                _, rms = out
            #_, rms = eval_rms(pred,actual,mesh,keys=pressure_vars,by_level=True,stdout=doprint,bbox=bbox)

        eval_shape = (B,N1,N2,nS)
        to_eval = lambda z : z[...,nPL:].view(eval_shape)
        pred = to_eval(y)
        actual = to_eval(yt)
        ret = eval_metric(pred,actual,weights,mesh,keys=sfc_vars,stdout=doprint,bbox=bbox,metric_fn=metric_fn, resolution=resolution, bylat=bylat)
        sfc_rms = ret[1]
        if bylat:
            byl.update(ret[2])
        if only_sfc:
            return sfc_rms
        else:
            rms.update(sfc_rms)
        #rms.update(eval_rms(pred,actual,mesh,keys=sfc_vars,stdout=doprint,bbox=bbox)[1])

        if bylat:
            return rms, byl

        return rms

# def compute_errors_sfc(preds, actuals, weights, metric_fns=['rmse', 'bias']):
#     assert preds.shape == (720,1440), f"Predictions shape {preds.shape} does not match expected (720,1440)"
#     assert actuals.shape == (720,1440), f"Actuals shape {actuals.shape} does not match expected (720,1440)"
#     assert (720/weights.shape[0]) % 1 == 0 and (1440/weights.shape[1]) % 1 == 0, f"Weights shape {weights.shape} does not match expected (120,240)"
#     ds = int(720/weights.shape[0]) # downsample factor
#     preds = preds[::ds,::ds]
#     actuals = actuals[::ds,::ds]
#     return eval_metrics_sfc(preds, actuals, weights, metric_fns)

def get_mesh_weights(resolution=0.25):
    assert resolution >= 0.25, f"Resolution {resolution} is not >= 0.25"
    assert (resolution/0.25) % 1 == 0, f"Resolution {resolution} is not a multiple of 0.25"
    assert 180 % resolution == 0, f"180 is not a multiple of resolution {resolution}"
    lats = np.arange(90, -90, -resolution) 
    lats = np.repeat(lats, 2*lats.shape[0]).reshape(lats.shape[0],2*lats.shape[0])
    weights = np.cos(lats * np.pi/180) 
    return weights

def unnorm_output(x,y,model,dt,y_is_deltas=None,skip_x=False,pad_x=False, skip_y=False):
    # this actually unnorms the output and the input together
    if not skip_y:
        x = x[...,:y.shape[-1]]
        # if not isinstance(y, TensorWithMetadata):
        #     x = x[...,:y.shape[-1]]
        # elif y.meta.delta_info is not None:
        #     if y_is_deltas is not None:
        #         assert y_is_deltas == True, f"y_is_deltas is {y_is_deltas} but y.meta.delta_info is {y.meta.delta_info}"
        #     assert y.meta.delta_info == dt, f'dt mismatch: {y.meta.delta_info} != {dt}'
        #     x,y = unnorm_output_partial(x,y,model,dt)
        # else: 
        #     x = x[...,:y.shape[-1]]
        y = unnorm(y,model.config.outputs[0])
        if skip_x:
            return None, y
    
    x = unnorm(x,model.config.inputs[0],pad_x=pad_x)
    return x,y

def unnorm_output_partial_notinplace(x,y,model,dt):
    B,N1,N2,O = y.shape
    xp = x[...,:O]
    #print("checksum for", str(dt), "is", dnormx.sum())
    #dnormx = dnorm.to(x.device)
    #if not isinstance(x,torch.Tensor):
    #    dnormx = dnormx.to('cpu').numpy()
    if model.config.output_deltas:
        dnormx = model.decoders[str(dt)].delta_norm_matrix.clone()
        yc = xp + y * dnormx
    else:
        yc = y
    return xp,yc


def unnorm_output_partial(x,y,model,dt):
    B,N1,N2,O = y.shape
    x = x[...,:O]
    if y.meta.delta_info is not None:
        assert y.meta.delta_info == dt, "dt mismatch"
        dnorm = model.decoders[str(dt)].delta_norm_matrix
        dnorm = dnorm.to(x.device)
        if not isinstance(x,torch.Tensor):
            dnorm = dnorm.to('cpu').numpy()
        y = x + y * dnorm
    return x,y

def unnorm(x,mesh,pad_x=False, exp_log_vars=False):
    if pad_x and x.shape[-1] < mesh.state_norm_stds.shape[-1]:
        x = torch.nn.functional.pad(x, (0,mesh.state_norm_stds.shape[-1]-x.shape[-1]), mode='constant', value=0)

    x = x*torch.Tensor(mesh.state_norm_stds).to(x.device) + torch.Tensor(mesh.state_norm_means).to(x.device)
    if exp_log_vars:
        for i, var in enumerate(mesh.sfc_vars):
            if var in log_vars:
                j = mesh.n_pr + i
                x[...,j] = torch.exp(x[...,j])
    assert x.dtype == torch.float32, "Bro you can't unnorm in fp16"
    return x

def unnorm_cpu(x,mesh):
    x = x*mesh.state_norm_stds + mesh.state_norm_means
    return x

