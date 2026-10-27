"""Notebook controls and selected-parameter fitting for the native KC simulator."""
import html
import math
import time

import ipywidgets as widgets
import matplotlib.pyplot as plt
import torch
from IPython.display import Audio, HTML, display, clear_output
from cels import STFTCumulativeEnergyLoss, STFTPower
from .kc_string_forward import ForwardString, PARAMETER_NAMES, nominal_parameters

LABELS = ('Density', 'Tension', 'Radius', "Young's modulus", 'Length', 'Excitation amplitude',
          'Frequency-independent damping', 'Frequency-dependent damping')
UNITS = ('kg/m³', 'N', 'm', 'Pa', 'm', 'm', '1/s', 'm²/s')
PERTURBATION = torch.tensor([1.1, .9, 1.1, .9, 1.1, .9, 1.1, .9], dtype=torch.float64)


def stability_score(physical, settings):
    density, tension, radius, modulus, length, _, _, sigma_1 = physical.detach().tolist()
    h = length / settings['intervals'] / 1.01
    dt = 1 / (settings['sample_rate'] * settings['oversampling'])
    mass = density * math.pi * radius**2
    return (tension / mass * dt**2 + 4*sigma_1*dt) / h**2 + modulus*radius**2/density*dt**2/h**4


def feasible(physical, settings):
    if not bool(torch.isfinite(physical).all()):
        return False
    if not bool((physical[:6] > 0).all()) or not bool((physical[6:] >= 0).all()):
        return False
    try:
        return stability_score(physical, settings) <= 1
    except (OverflowError, ZeroDivisionError):
        return False


class Renderer:
    """Fixed-grid native simulation and, when requested, filtered decimation."""
    def __init__(self, settings):
        self.settings = dict(settings)
        self.factor = settings['oversampling']
        self.model = ForwardString(settings['intervals'], settings['sample_rate'] * self.factor)
        self.samples = round(settings['duration'] * settings['sample_rate'])
        self.filter = None
        if self.factor > 1:
            self.half_width = 16 * self.factor
            index = torch.arange(-self.half_width, self.half_width + 1, dtype=torch.float64)
            cutoff = .45 / self.factor
            kernel = 2*cutoff*torch.sinc(2*cutoff*index) * torch.hann_window(index.numel(), periodic=False, dtype=torch.float64)
            self.filter = (kernel/kernel.sum()).reshape(1, 1, -1)

    def __call__(self, theta):
        audio = self.model(theta, self.settings['duration'])
        if self.filter is not None:
            padded = torch.nn.functional.pad(audio.reshape(1, 1, -1),
                                            (self.half_width, self.half_width), mode='replicate')
            audio = torch.nn.functional.conv1d(padded, self.filter, stride=self.factor).flatten()
        return audio[:self.samples]


def fit_selected(renderer, criterion, initial, mask, optimizer_name, learning_rate, max_steps, progress=None):
    """Fit selected entries only. Both methods start from the same initial vector."""
    nominal = nominal_parameters()
    initial_theta = initial / nominal
    indices = torch.tensor([i for i, selected in enumerate(mask) if selected], dtype=torch.long)
    if not indices.numel():
        with torch.no_grad():
            initial_loss = criterion(renderer(initial_theta)).item()
        return dict(name=optimizer_name, best_physical=initial.clone(), best_loss=initial_loss,
                    history=[dict(iteration=0, loss=initial_loss, best_loss=initial_loss, evaluations=1, seconds=0.)],
                    iterations=0, evaluations=1, rejected_trials=0, seconds=0.,
                    stop_reason='No parameters selected; the prediction equals the initial signal.', mask=list(mask))
    # Shift damping coordinates so a zero initial damping value is allowed.
    # The other six coordinates are ordinary log multipliers.
    offset = (indices >= 6).to(torch.float64)
    z = torch.nn.Parameter((initial_theta[indices] + offset).log())
    def theta_from(values):
        return initial_theta.scatter(0, indices, values.exp() - offset)
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam([z], lr=learning_rate)
    else:
        optimizer = torch.optim.LBFGS([z], lr=learning_rate, max_iter=1, max_eval=25,
                    history_size=20, line_search_fn='strong_wolfe', tolerance_grad=1e-12, tolerance_change=1e-14)
    started = time.perf_counter()
    calls = rejected = 0
    history = []
    best_loss = float('inf')
    best_physical = initial.clone()
    cache = {}

    def evaluate():
        nonlocal calls, rejected, cache
        calls += 1
        optimizer.zero_grad()
        theta = theta_from(z)
        valid = feasible(theta.detach()*nominal, renderer.settings)
        if valid:
            audio = renderer(theta)
            valid = bool(torch.isfinite(audio).all())
        if valid:
            value = criterion(audio)
        else:
            rejected += 1
            # Invalid-domain trials never run the unstable string or enter the
            # history as a valid CeL improvement. Line search can backtrack.
            value = 1e6 + z.square().sum()
        value.backward()
        if not bool(torch.isfinite(value)) or not bool(torch.isfinite(z.grad).all()):
            raise ValueError('Nonfinite loss/gradient. Reduce excitation or choose safer parameters.')
        cache = dict(z=z.detach().clone(), loss=value.item(), valid=valid)
        return value

    def record(iteration):
        nonlocal best_loss, best_physical
        if cache['valid'] and cache['loss'] < best_loss:
            best_loss = cache['loss']
            best_physical = (theta_from(z).detach()*nominal)
        history.append(dict(iteration=iteration, loss=cache['loss'], best_loss=best_loss,
                            evaluations=calls, seconds=time.perf_counter()-started))
        if progress and (iteration % 10 == 0):
            progress(history[-1])

    evaluate()
    record(0)
    reason = 'Reached the maximum step count.'
    if best_loss == 0:
        reason = 'Initial signal already has zero loss.'
    else:
        for iteration in range(1, max_steps+1):
            if optimizer_name == 'adam':
                old = z.detach().clone()
                optimizer.step()
                proposal = z.detach().clone()
                # Nonnegative damping may start exactly at its boundary.
                proposal = torch.where(offset > 0, proposal.clamp_min(0), proposal)
                fraction = 1.
                while not feasible(theta_from(old+fraction*(proposal-old)).detach()*nominal, renderer.settings):
                    rejected += 1
                    fraction *= .5
                    if fraction < 2**-30:
                        raise ValueError('Cannot find a feasible Adam proposal; reduce its learning rate.')
                with torch.no_grad():
                    z.copy_(old+fraction*(proposal-old))
                evaluate()
            else:
                optimizer.step(evaluate)
                if not torch.equal(cache['z'], z.detach()):
                    evaluate()
                if not cache['valid']:
                    raise ValueError('L-BFGS accepted an invalid trial. Reduce its learning rate.')
            record(iteration)
            if len(history) > 20:
                before = history[-21]['best_loss']
                improvement = (before-best_loss)/max(before, 1e-30)
                if improvement < .1:
                    reason = 'Less than 10% best-loss improvement over the last 20 optimizer steps.'
                    break
    # Use exactly the original physical entries for frozen parameters.
    best_physical = torch.where(torch.tensor(mask), best_physical, initial)
    return dict(name=optimizer_name, best_physical=best_physical, best_loss=best_loss,
                history=history, iterations=len(history)-1, evaluations=calls, rejected_trials=rejected,
                seconds=time.perf_counter()-started, stop_reason=reason, mask=list(mask))


class OptimizationWorkbench:
    def __init__(self):
        self.target = self.initial = self.renderer = self.criterion = None
        self.results = {}
        self.busy = False
        nominal = nominal_parameters().tolist()
        self.target_fields = [widgets.FloatText(value=v, layout=widgets.Layout(width='180px')) for v in nominal]
        self.initial_fields = [widgets.FloatText(value=v, layout=widgets.Layout(width='180px')) for v in (nominal_parameters()*PERTURBATION).tolist()]
        self.optimize_boxes = [widgets.Checkbox(value=True, indent=False, layout=widgets.Layout(width='90px')) for _ in nominal]
        self.sample_rate = widgets.BoundedIntText(value=44100, min=1000, max=384000, description='Audio rate', style={'description_width':'130px'})
        self.duration = widgets.BoundedFloatText(value=4., min=.001, max=60., description='Duration (s)', style={'description_width':'130px'})
        self.intervals = widgets.BoundedIntText(value=128, min=4, max=4096, description='Grid intervals', style={'description_width':'130px'})
        self.oversampling = widgets.BoundedIntText(value=1, min=1, max=16, description='Oversampling', style={'description_width':'130px'})
        self.target_output, self.initial_output = widgets.Output(), widgets.Output()
        self.run_outputs = {name:widgets.Output() for name in ('adam','lbfgs')}
        self.prediction_outputs = {name:widgets.Output() for name in ('adam','lbfgs')}
        self.learning_rates = {name:widgets.FloatText(value=value, description='Learning rate') for name,value in [('adam',.01),('lbfgs',1.)]}
        self.max_steps = {name:widgets.BoundedIntText(value=500,min=1,max=10000,description='Max steps') for name in ('adam','lbfgs')}
        self.buttons = []
        for field in self.target_fields + [self.sample_rate,self.duration,self.intervals,self.oversampling]:
            field.observe(self._target_changed, names='value')
        for field in self.initial_fields:
            field.observe(self._initial_changed, names='value')
        for field in self.optimize_boxes:
            field.observe(lambda _: self._clear_results(), names='value')

    def _button(self, description, callback, style=''):
        button = widgets.Button(description=description, button_style=style, layout=widgets.Layout(width='auto'))
        button.on_click(callback)
        self.buttons.append(button)
        return button

    def _message(self, output, message):
        with output:
            clear_output(wait=False)
            print(message)

    def _clear_results(self):
        self.results.clear()
        for output in self.prediction_outputs.values():
            self._message(output, 'No current prediction. Run the optimizer after generating both sounds.')
        for output in self.run_outputs.values():
            self._message(output, 'Ready after generating the target and initial signal.')

    def _target_changed(self, _):
        self.target = self.initial = self.criterion = self.renderer = None
        self._clear_results()
        self._message(self.target_output, 'Target controls changed: generate the target again.')
        self._message(self.initial_output, 'Generate the target, then the initial signal again.')

    def _initial_changed(self, _):
        self.initial = None
        self._clear_results()
        self._message(self.initial_output, 'Initial controls changed: generate the initial signal again.')

    def _rows(self, fields, with_mask=False):
        rows = [widgets.HBox([widgets.HTML('<b>Parameter</b>',layout=widgets.Layout(width='240px')),
                             widgets.HTML('<b>Physical value</b>',layout=widgets.Layout(width='180px')),
                             widgets.HTML('<b>Units</b>',layout=widgets.Layout(width='70px'))] +
                            ([widgets.HTML('<b>Optimise?</b>')] if with_mask else []))]
        for i in range(8):
            cells = [widgets.Label(LABELS[i], layout=widgets.Layout(width='240px')), fields[i],
                     widgets.Label(UNITS[i], layout=widgets.Layout(width='70px'))]
            if with_mask: cells.append(self.optimize_boxes[i])
            rows.append(widgets.HBox(cells))
        return rows

    def target_panel(self):
        def select(value):
            for checkbox in self.optimize_boxes: checkbox.value=value
        return widgets.VBox([widgets.HBox([self.sample_rate,self.duration]),
            widgets.HBox([self.intervals,self.oversampling]),*self._rows(self.target_fields,True),
            widgets.HBox([self._button('Select all',lambda _:select(True)),self._button('Select none',lambda _:select(False)),
                          self._button('Generate target',lambda _:self.generate_target(),'primary')]),self.target_output])

    def initial_panel(self):
        def fill(perturb):
            values=torch.tensor([f.value for f in self.target_fields],dtype=torch.float64)
            if perturb: values=values*PERTURBATION
            for field,value in zip(self.initial_fields,values.tolist()): field.value=value
        return widgets.VBox([widgets.HTML('Uses the generated target’s audio rate, duration, grid and oversampling.'),
            *self._rows(self.initial_fields),widgets.HBox([
                self._button('Copy target parameters',lambda _:fill(False)),
                self._button('Set signed 10% perturbation',lambda _:fill(True)),
                self._button('Generate initial signal',lambda _:self.generate_initial(),'primary')]),self.initial_output])

    def optimizer_panel(self, name):
        return widgets.VBox([widgets.HBox([self.learning_rates[name],self.max_steps[name]]),
            widgets.HTML('Patience: stop if best loss improves by &lt;10% over 20 completed optimizer steps. '
                         'L-BFGS line-search evaluations are counted separately.'),
            self._button(f'Run {name.upper()} from initial signal',lambda _:self.run(name),'success'),self.run_outputs[name]])

    def prediction_panel(self, name):
        return widgets.VBox([self._button('Show / refresh best prediction',lambda _:self.show_prediction(name)),self.prediction_outputs[name]])

    def _validate(self, physical, settings):
        if settings['intervals']%2: raise ValueError('Grid intervals must be even.')
        if round(settings['sample_rate']*settings['duration'])<1024:
            raise ValueError('The default loss needs at least 1024 audio samples. Increase the duration.')
        if not feasible(physical,settings):
            raise ValueError('Invalid physical parameters or unstable grid. Use positive values (damping may be zero), '
                             'reduce grid intervals, or increase the integration rate/oversampling. No values were changed.')

    def generate_target(self):
        try:
            settings=dict(sample_rate=self.sample_rate.value,duration=self.duration.value,
                          intervals=self.intervals.value,oversampling=self.oversampling.value)
            physical=torch.tensor([f.value for f in self.target_fields],dtype=torch.float64)
            self._validate(physical,settings)
            renderer=Renderer(settings)
            with torch.no_grad(): audio=renderer(physical/nominal_parameters())
            if not bool(torch.isfinite(audio).all()): raise ValueError('Target simulation was not finite.')
            criterion=STFTCumulativeEnergyLoss(sample_rate=settings['sample_rate']).bind(audio)
            self.renderer,self.criterion=renderer,criterion
            self.target=dict(physical=physical,audio=audio,settings=settings)
            self.initial=None
            self._clear_results()
            self._refresh_audio()
            self._message(self.initial_output,'Target generated. Generate the initial signal next.')
        except Exception as error:
            self._message(self.target_output,f'Target not generated: {error}')

    def generate_initial(self):
        try:
            if self.target is None: raise ValueError('Generate the target first.')
            physical=torch.tensor([f.value for f in self.initial_fields],dtype=torch.float64)
            self._validate(physical,self.target['settings'])
            with torch.no_grad(): audio=self.renderer(physical/nominal_parameters())
            if not bool(torch.isfinite(audio).all()): raise ValueError('Initial simulation was not finite.')
            loss=self.criterion(audio).item()
            self.initial=dict(physical=physical,audio=audio,loss=loss)
            self._clear_results()
            self._refresh_audio()
        except Exception as error:
            self._message(self.initial_output,f'Initial signal not generated: {error}')

    def _gain(self):
        signals=[self.target['audio']]
        if self.initial is not None: signals.append(self.initial['audio'])
        signals += [r['audio'] for r in self.results.values()]
        return .9/max(max(s.abs().max().item() for s in signals),1e-30)

    def _refresh_audio(self):
        gain=self._gain()
        for name,signal,output in [('Target',self.target,self.target_output),('Initial',self.initial,self.initial_output)]:
            if signal is None: continue
            with output:
                clear_output(wait=True)
                print(f"{name}: {self.target['settings']['duration']:g} s at {self.target['settings']['sample_rate']:,} Hz. "
                      f"Stability score {stability_score(signal['physical'],self.target['settings']):.4f} ≤ 1.")
                if name=='Initial': print(f"Default CeL loss: {signal['loss']:.8g}")
                print(f'Shared playback gain: {gain:.3g} (loss uses unscaled audio).')
                display(Audio((signal['audio']*gain).numpy(),rate=self.target['settings']['sample_rate'],normalize=False))

    def run(self,name):
        output=self.run_outputs[name]
        if self.busy: return
        controls=self.target_fields+self.initial_fields+self.optimize_boxes+[self.sample_rate,self.duration,self.intervals,self.oversampling]+list(self.learning_rates.values())+list(self.max_steps.values())+self.buttons
        try:
            if self.target is None or self.initial is None: raise ValueError('Generate the target and initial signal first.')
            rate=self.learning_rates[name].value
            if not math.isfinite(rate) or rate<=0: raise ValueError('Learning rate must be finite and positive.')
            self.busy=True
            for control in controls: control.disabled=True
            with output:
                clear_output(wait=True)
                print(f'{name.upper()} running from the saved initial signal…',flush=True)
                def progress(row):
                    print(f"Step {row['iteration']:4d} | best CeL {row['best_loss']:.6g} | evaluations {row['evaluations']}",flush=True)
                result=fit_selected(self.renderer,self.criterion,self.initial['physical'],
                    [c.value for c in self.optimize_boxes],name,rate,self.max_steps[name].value,progress)
                with torch.no_grad(): result['audio']=self.renderer(result['best_physical']/nominal_parameters())
                result['verified_loss']=self.criterion(result['audio']).item()
                if not math.isclose(result['verified_loss'],result['best_loss'],rel_tol=1e-6,abs_tol=1e-10):
                    raise ValueError('Best-checkpoint verification failed.')
                self.results[name]=result
                print(f"Finished: {result['iterations']} steps, {result['evaluations']} evaluations, {result['seconds']:.1f} s.")
                print(result['stop_reason'])
            self._refresh_audio()
            for key in self.results: self.show_prediction(key)
        except Exception as error:
            with output: print(f'Run not completed: {error}')
        finally:
            self.busy=False
            for control in controls: control.disabled=False

    def show_prediction(self,name):
        output=self.prediction_outputs[name]
        with output:
            clear_output(wait=True)
            if name not in self.results:
                print('Run this optimizer first. Predictions update automatically when it finishes.')
                return
            result=self.results[name]
            print(f"{name.upper()} best CeL: {result['best_loss']:.8g}. {result['stop_reason']}")
            display(Audio((result['audio']*self._gain()).numpy(),rate=self.target['settings']['sample_rate'],normalize=False))
            rows=''.join('<tr>'+''.join(f'<td style="padding:4px 12px">{value}</td>' for value in
                [html.escape(LABELS[i]),f'{self.target["physical"][i]:.7g}',f'{self.initial["physical"][i]:.7g}',
                 f'{result["best_physical"][i]:.7g}','Yes' if result['mask'][i] else 'Fixed'])+'</tr>' for i in range(8))
            display(HTML('<table><tr><th>Parameter</th><th>Target</th><th>Initial</th><th>Prediction</th><th>Optimised</th></tr>'+rows+'</table>'))
            fig,axes=plt.subplots(1,2,figsize=(11,3.3),layout='constrained')
            for key,r in self.results.items():
                h=r['history']
                axes[0].semilogy([x['iteration'] for x in h],[max(x['best_loss'],1e-16) for x in h],label=key.upper())
                axes[1].semilogy([x['seconds'] for x in h],[max(x['best_loss'],1e-16) for x in h],label=key.upper())
            for axis,xlabel in zip(axes,['Optimizer steps','Seconds']):
                axis.set(xlabel=xlabel,ylabel='Best default CeL loss');axis.legend();axis.grid(alpha=.2)
            display(fig);plt.close(fig)
            transform=STFTPower(sample_rate=self.target['settings']['sample_rate'])
            with torch.no_grad():
                powers=[transform(s) for s in [self.target['audio'],self.initial['audio'],result['audio']]]
            reference_peak=max(p.max().item() for p in powers)
            frequency,times=transform.coordinates(powers[0]);times=times+512/self.target['settings']['sample_rate']
            fig,axes=plt.subplots(1,3,figsize=(13,3.5),sharex=True,sharey=True,layout='constrained')
            for axis,power,title in zip(axes,powers,['Target','Initial',name.upper()+' best prediction']):
                db=10*torch.log10((power/reference_peak).clamp_min(1e-8))
                artist=axis.pcolormesh(times.numpy(),frequency.numpy(),db.numpy(),shading='auto',vmin=-80,vmax=0,cmap='magma',rasterized=True)
                axis.set(title=title,xlabel='Time (s)',ylim=(0,min(2000,self.target['settings']['sample_rate']/2)))
            axes[0].set_ylabel('Frequency (Hz)')
            fig.colorbar(artist,ax=axes,label='Power (dB, shared reference)')
            display(fig);plt.close(fig)
