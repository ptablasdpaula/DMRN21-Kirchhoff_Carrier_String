#!/usr/bin/env python3
"""Exact damped linear poles of the matched N=176 FDTD at 34x or 55x.

Idealized fretted scenarios are derived from Woodhouse et al. Table I.  This
examines poles, not the nonlinear response, GPU implementation, or audibility.
"""
import argparse
import cmath
import csv
import gzip
import math
from pathlib import Path
import guitar_modal_dispersion as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmarks/guitar_damped_pole_results'
N = 176
FS = 44100 * 34
SIGMA0 = 1.0
SIGMA1 = 1e-4
L_OPEN = 0.65
LIMIT = 20000
NYQUIST = 44100/2


def roots(n, f0, b, length):
    """Roots of the actual linearized FDTD recurrence, with damping."""
    k = 1/FS
    eig = base.spatial_lambda(n)/length**2
    w = base.omega(n, f0, b)
    a = (2-k*k*w*w-2*SIGMA1*k*eig)/(1+SIGMA0*k)
    q = (SIGMA0*k-1+2*SIGMA1*k*eig)/(1+SIGMA0*k)
    d = cmath.sqrt(a*a+4*q)
    return (a+d)/2, (a-d)/2


def continuous(n, f0, b, length):
    """Damped continuum angular-frequency pole, or None if overdamped."""
    loss = SIGMA0 + SIGMA1*(math.pi*n/length)**2
    w = 2*math.pi*base.frequency(n, f0, b)
    return math.sqrt(w*w-loss*loss)/(2*math.pi) if w > loss else None


def write(path, rows):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)


def main():
    global FS, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--oversampling', type=int, choices=(34,55), default=34)
    args = parser.parse_args()
    FS = 44100 * args.oversampling
    if args.oversampling != 34:
        OUT = ROOT / f'benchmarks/guitar_damped_pole_results_{args.oversampling}x'
    OUT.mkdir(parents=True, exist_ok=True)
    modes=[]; summaries=[]
    for material, bs in base.OPEN_B.items():
        for string, f_open, micro in zip(base.STRINGS, base.OPEN_F1, bs):
            for fret in range(25):
                scale=2**(fret/12); b_open=micro*1e-6
                b=b_open*scale**2; f0=f_open/math.sqrt(1+b_open)*scale
                f1=f0*math.sqrt(1+b); length=L_OPEN/scale
                case=f'{material}_{string}_fret_{fret}'
                radius=max(abs(r) for n in range(1,N) for r in roots(n,f0,b,length))
                stable=radius < 1
                case_modes=[]; n=1
                # For this case, the squared damped continuum frequency is a
                # positive-coefficient polynomial in n² (apart from a constant),
                # so it is strictly increasing on positive integers.
                coeff2 = f0*f0-SIGMA0*SIGMA1/(2*length**2)
                coeff4 = f0*f0*b-SIGMA1**2*math.pi**2/(4*length**4)
                if coeff2 <= 0 or coeff4 <= 0 or continuous(1,f0,b,length) is None:
                    raise ValueError(f'{case}: damped-mode monotonicity assumption failed')
                while continuous(n,f0,b,length) <= LIMIT:
                    fc=continuous(n,f0,b,length)
                    if fc is None: status,fd,cents='continuum_overdamped',None,None
                    elif n>=N: status,fd,cents='not_represented',None,None
                    elif not stable: status,fd,cents='full_grid_unstable',None,None
                    else:
                        r=roots(n,f0,b,length)[0]
                        if not r.imag: status,fd,cents='discrete_overdamped',None,None
                        else:
                            fd=cmath.phase(r)*FS/(2*math.pi)
                            cents=1200*math.log2(fd/fc)
                            status='represented' if fd<NYQUIST else 'output_nyquist_exceeded'
                    case_modes.append(dict(case=case,material=material,string=string,fret=fret,
                        length_m=length,first_partial_undamped_hz=f1,tension_f0_hz=f0,B=b,
                        full_grid_stable=stable,max_pole_radius=radius,mode=n,
                        continuous_damped_hz=fc,discrete_damped_hz=fd,error_cents=cents,status=status))
                    n+=1
                valid=[r for r in case_modes if r['status']=='represented']
                entry=dict(case=case,material=material,string=string,fret=fret,length_m=length,
                    first_partial_undamped_hz=f1,tension_f0_hz=f0,B=b,full_grid_stable=stable,
                    max_pole_radius=radius,continuous_modes_to_20khz=len(case_modes),
                    missing_modes_to_20khz=sum(r['status']=='not_represented' for r in case_modes),
                    max_abs_cents=max((abs(r['error_cents']) for r in valid),default=None))
                for mode in (1,3,5,10,20,50,100,175):
                    row=next((r for r in valid if r['mode']==mode),None)
                    entry[f'mode_{mode}_error_cents']=row['error_cents'] if row else None
                summaries.append(entry);modes.extend(case_modes)
    write(OUT/'cases.csv',summaries);write(OUT/'modes.csv.gz',modes)
    print('Scenarios',len(summaries),'stable',sum(r['full_grid_stable'] for r in summaries),
          'unstable',sum(not r['full_grid_stable'] for r in summaries))
    for mode in (1,3,5,10,20,50,100,175):
        subset=[r for r in modes if r['mode']==mode and r['status']=='represented']
        worst=max(subset,key=lambda r:abs(r['error_cents']))
        print('mode',mode,'cases',len(subset),'max abs cents',abs(worst['error_cents']),
              'case',worst['case'],'continuum',worst['continuous_damped_hz'],
              'discrete',worst['discrete_damped_hz'])


if __name__ == '__main__':
    main()
