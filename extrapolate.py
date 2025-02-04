from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime
from scipy.ndimage import gaussian_filter1d as gf


def power_law(x, a, k):
    return a * np.power((x), -k)

def power_law2(x, a, k, b):
    return a * np.power((x+b), -k)

def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def logistic(x, a, b, k):
    return a / (1 + b * np.exp(-k * x))

def linear(x, a, b):
    return a * x + b

def extrap_error(steps,error,st,end,to,fit_fn,fit_fn2 = None):
    steps = np.array(steps)
    idx = np.where((st < steps) & (steps < end))[0]
    steps = steps[idx]
    error = np.array(error)[idx]
    params, _ = curve_fit(fit_fn, steps, error)
    if fit_fn2 is not None:
        fit_fn = fit_fn2
        params, _ = curve_fit(fit_fn2, steps, error,p0=[*params,0])
    steps_e = np.arange(st, to, 20)
    fit_e = fit_fn(steps_e, *params)
    return steps_e, fit_e, params

def extrap_plot(steps,error,st,end,to,fit_fn,fit_fn2 = None,c='C1'):
    x,y,p = extrap_error(steps,error,st,end,to,fit_fn,fit_fn2=fit_fn2)
    plt.plot(x,y,color=c)
    if fit_fn2 is not None:
        fit_fn = fit_fn2
    e = fit_fn(np.array([st,end]),*p)
    plt.plot([st,end],e,'*',color=c)

def predict_training(log_file,var,step_start,step_end,step_extrap):
    log_file = f"/fast/ignored/runs/{log_file}" #this is actually a dir
    acc = EventAccumulator(log_file)
    acc.Reload()
    d = acc.Scalars(var)
    e = [x.value for x in d]
    steps = [x.step for x in d]
    idx = np.where(np.array(steps) > step_start)[0]
    steps = np.array(steps)[idx]
    error = np.array(e)[idx]

    plt.figure(figsize=(18,6))
    plt.plot(steps, error,c='C0',alpha=0.3)

    pf = power_law2 ; pf = None
    extrap_plot(steps,error,step_start,step_end,step_extrap,power_law,pf,c='C1')
    th = 0.55 ; end = int(step_start*th+step_end*(1-th))
    extrap_plot(steps,error,step_start,end,step_extrap,power_law,pf,c='C2')
    th = 0.45 ; start = int(step_start*th+step_end*(1-th))
    extrap_plot(steps,error,start,step_end,step_extrap,power_law,pf,c='C3')
    plt.plot(steps, gf(error,10),c='C0',alpha=1)
    #extrap_plot(steps,error,7_000,11_000,step_extrap,linear,c='C2')

    now = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    name = log_file.split('/')[-1]
    plt.title(f"{name}:{var} | Extpolated at {now}")
    plt.grid()
    plt.savefig(f'ignored/plots/{name}.png')

predict_training("run_Oct2-3dDelta-Cos_20231002-125439",
                 'Error/129_z',
                 500,
                 3_000,
                 200_000)


if 0:
    predict_training("run_Oct1-3dfix2_20231001-143508",
                 'Error/129_z',
                 2_000,
                 13_000,
                 200_000)

if 0:
    predict_training("run_Oct1-Conv-only_20231001-183117",
                    'Error/129_z',
                    2_000,
                    10_000,
                    200_000)
