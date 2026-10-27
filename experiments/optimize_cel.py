"""Fit one fixed-grid KC target with Adam and L-BFGS using upstream default CeL.

Run: pixi run python experiments/optimize_cel.py
Stop when best loss improves by <10% over 20 completed optimizer steps.
L-BFGS uses max_iter=1; its line-search calls are counted separately.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import wave

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kc_string_forward import ForwardString, nominal_parameters, PARAMETER_NAMES
from cels import STFTCumulativeEnergyLoss

LOSS_COMMIT = '9fdbd60f5b67def8dd8884a006541e0ca4dbbf0d'
INITIAL = [1.1, .9, 1.1, .9, 1.1, .9, 1.1, .9]


def stability_score(theta):
    density, tension, radius, modulus, length, amplitude, sigma_0, sigma_1 = (nominal_parameters()*theta).tolist()
    mass = density*math.pi*radius**2
    c_squared = tension/mass
    K_squared = modulus*radius**2/(4*density)
    safe_spacing = (length/128)/1.01
    dt = 1/44100
    return (c_squared*dt**2 + 4*sigma_1*dt)/safe_spacing**2 + 4*K_squared*dt**2/safe_spacing**4


def should_stop(history, window=20, minimum_improvement=.1):
    if len(history) <= window:
        return False, None
    before = history[-window-1]['best_loss']
    improvement = (before-history[-1]['best_loss']) / max(before, 1e-30)
    return improvement < minimum_improvement, improvement


def fit(name, model, criterion, duration, max_steps, output_path):
    # Log multipliers enforce positive physical quantities without clipping.
    z = torch.nn.Parameter(torch.tensor(INITIAL,dtype=torch.float64).log())
    if name == 'adam':
        optimizer = torch.optim.Adam([z],lr=.01)
    else:
        optimizer = torch.optim.LBFGS([z],lr=1.,max_iter=1,max_eval=25,
            history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-12,tolerance_change=1e-14)
    history=[]
    calls=0
    rejected=0
    best_loss=float('inf')
    best_theta=None
    cache={}
    started=time.perf_counter()

    def evaluate(backward=True):
        nonlocal calls,rejected,cache
        calls+=1
        optimizer.zero_grad()
        theta=z.exp()
        # This guard is a feasibility rejection, not a regularizer on valid fits.
        valid = bool(torch.isfinite(z).all()) and bool((z.detach().abs() < math.log(4)).all())
        score=stability_score(theta.detach()) if valid else float('inf')
        valid = valid and score <= 1
        if not valid:
            rejected+=1
            value=1e6+z.square().sum()
        else:
            audio=model(theta,duration)
            if not bool(torch.isfinite(audio).all()):
                rejected+=1
                valid=False
                value=1e6+z.square().sum()
            else:
                value=criterion(audio)
        if backward:
            value.backward()
            if not bool(torch.isfinite(z.grad).all()):
                raise RuntimeError('Nonfinite gradient')
        cache=dict(z=z.detach().clone(),loss=value.item(),valid=valid,stability_score=score)
        return value

    def record(iteration):
        nonlocal best_loss,best_theta
        current=cache['loss']
        if cache['valid'] and current < best_loss:
            best_loss=current
            best_theta=z.detach().exp().clone()
        row=dict(iteration=iteration,loss=current,best_loss=best_loss,
                 theta=z.detach().exp().tolist(),evaluations=calls,rejected_trials=rejected,
                 elapsed_seconds=time.perf_counter()-started,stability_score=cache['stability_score'])
        history.append(row)
        output_path.write_text(json.dumps(history,indent=2)+'\n')
        if iteration % 10 == 0:
            print(name,iteration,'loss',current,'best',best_loss,'evaluations',calls,flush=True)

    evaluate()
    record(0)
    stopping='maximum_steps'
    improvement=None
    for iteration in range(1,max_steps+1):
        if name=='adam':
            old_z=z.detach().clone()
            optimizer.step()
            # Feasible backtracking on the Adam proposal; preserve its direction.
            proposed=z.detach().clone()
            fraction=1.
            while True:
                candidate=old_z+fraction*(proposed-old_z)
                valid=bool((candidate.abs()<math.log(4)).all()) and stability_score(candidate.exp()) <= 1
                if valid:
                    with torch.no_grad(): z.copy_(candidate)
                    break
                rejected+=1
                fraction*=.5
                if fraction<2**-30:
                    raise RuntimeError('Unable to find a feasible Adam step')
            evaluate()
        else:
            optimizer.step(evaluate)
            if not torch.equal(cache['z'],z.detach()):
                evaluate()
            if not cache['valid']:
                raise RuntimeError('L-BFGS accepted an infeasible point')
        record(iteration)
        stop,improvement=should_stop(history)
        if stop:
            stopping='less_than_10_percent_improvement_over_20_steps'
            break
    return dict(optimizer=name,iterations=len(history)-1,evaluations=calls,rejected_trials=rejected,
                elapsed_seconds=time.perf_counter()-started,initial_loss=history[0]['loss'],
                best_loss=best_loss,best_theta=best_theta.tolist(),final_theta=z.detach().exp().tolist(),
                stop_reason=stopping,last_20_step_relative_improvement=improvement,history=history)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration',type=float,default=4.)
    parser.add_argument('--max-steps',type=int,default=500)
    parser.add_argument('--output',type=Path,default=ROOT/'.notebook-build/optimization_cel')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    model=ForwardString(intervals=128,sample_rate=44100)
    target_theta=torch.ones(8,dtype=torch.float64)
    with torch.no_grad():
        target=model(target_theta,args.duration)
        initial=model(torch.tensor(INITIAL,dtype=torch.float64),args.duration)
    loss=STFTCumulativeEnergyLoss(sample_rate=44100)
    criterion=loss.bind(target)
    assert criterion(target).item() == 0
    assert abs(criterion(initial).item()-loss(initial,target).item())<1e-12
    results={}
    for name in ('adam','lbfgs'):
        results[name]=fit(name,model,criterion,args.duration,args.max_steps,args.output/f'{name}_trace.json')
        (args.output/f'{name}_result.json').write_text(json.dumps(results[name],indent=2)+'\n')
    signals={'target':target,'initial':initial}
    for name,result in results.items():
        with torch.no_grad():
            fitted=model(torch.tensor(result['best_theta'],dtype=torch.float64),args.duration)
        signals[name]=fitted
        result['verified_best_loss']=criterion(fitted).item()
        assert abs(result['best_loss']-result['verified_best_loss'])<1e-12
        result['waveform_relative_l2_error']=((fitted-target).norm()/target.norm()).item()
        result['physical_parameters']=(nominal_parameters()*torch.tensor(result['best_theta'],dtype=torch.float64)).tolist()
    data=dict(configuration=dict(duration_seconds=args.duration,sample_rate=44100,intervals=128,
        dtype='float64',threads=1,parameter_names=PARAMETER_NAMES,nominal_parameters=nominal_parameters().tolist(),
        target_theta=target_theta.tolist(),initial_theta=INITIAL,
        initial_stability_score=stability_score(torch.tensor(INITIAL,dtype=torch.float64)),
        excitation='Fixed first sine mode; no pluck-position parameter',observation='Fixed midpoint',
        loss='STFTCumulativeEnergyLoss(sample_rate=44100), all other settings at library defaults',
        loss_repository='https://github.com/ptablasdpaula/ICASSP27-Phrase/tree/cel-library',loss_commit=LOSS_COMMIT,
        patience_window=20,minimum_relative_improvement=.1,
        feasible_region='Positive relative parameters between 0.25 and 4, plus fixed-grid linear stability score <= 1 with 1% spacing margin; infeasible trials rejected before simulation.',
        adam=dict(lr=.01),lbfgs=dict(lr=1.,max_iter=1,max_eval=25,history_size=20,line_search_fn='strong_wolfe')),
        results=results)
    (ROOT/'experiments/optimization_cel_results.json').write_text(json.dumps(data,indent=2)+'\n')
    # All files use one common gain, preserving relative amplitude differences.
    gain=.9/max(audio.abs().max().item() for audio in signals.values())
    for name,audio in signals.items():
        pcm=(audio*gain*32767).round().to(torch.int16).numpy().astype('<i2').tobytes()
        with wave.open(str(args.output/f'{name}.wav'),'wb') as handle:
            handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(44100); handle.writeframes(pcm)
    from experiments.plot_optimization_cel import create_plots
    create_plots(data,args.output,signals)
    print(json.dumps({name:{k:v for k,v in r.items() if k!='history'} for name,r in results.items()},indent=2),flush=True)


if __name__=='__main__':
    main()
