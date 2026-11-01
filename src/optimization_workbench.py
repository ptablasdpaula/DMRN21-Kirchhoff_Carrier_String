"""Shared notebook interface: ranges → fixed grid → target/initial → fitting."""
import html
import math
from pathlib import Path
import ipywidgets as widgets
import matplotlib.pyplot as plt
import torch
from IPython.display import Audio, HTML, display, clear_output
from cels import STFTPower
from .parameters import ParameterBox, MaterialParameterBox, perceptual_from_material
from .grid import GridConfig
from .synthesis import Renderer, modal_diagnostics, nonlinear_strength
from .optimization import fit, criterion_for, save_result

PERCEPTUAL_LABELS=('Taut fundamental f₀','Inharmonicity B','Amplitude','Radius','Uniform loss σ₀','Frequency-dependent loss σ₁')

MATERIAL_LABELS=('Density','Tension','Radius',"Young’s modulus",'Length','Amplitude','Uniform loss σ₀','Frequency-dependent loss σ₁')


class OptimizationWorkbench:
    def __init__(self, coordinate_system="perceptual"):
        self.grid=self.renderer=self.target=self.initial=self.criterion=None
        self.results={};self.busy=False;self.preparing=False;self.buttons=[]
        self.coordinates=widgets.Dropdown(options=[('Direct material parameters','material'),('Fundamental / inharmonicity','perceptual')],
            value=coordinate_system,description='Coordinates',layout=widgets.Layout(width='440px'))
        self.device=widgets.Dropdown(options=['cpu']+(['cuda'] if torch.cuda.is_available() else []),value='cpu',description='Device')
        self.backend=widgets.Dropdown(options=[('Original / native','native'),('Fused spatial','spatial'),('Full-grid modal','modal')],value='native',description='Solver')
        box=MaterialParameterBox() if coordinate_system=='material' else ParameterBox()
        self.labels=MATERIAL_LABELS if coordinate_system=='material' else PERCEPTUAL_LABELS
        self.names,self.units=box.names,box.units
        self.range_fields=[(widgets.FloatText(value=low,layout=widgets.Layout(width='170px')),
                            widgets.FloatText(value=high,layout=widgets.Layout(width='170px'))) for low,high in box.ranges]
        self.range_rows=widgets.VBox()
        self.sample_rate=widgets.BoundedIntText(value=44100,min=1000,max=384000,description='Output rate')
        self.duration=widgets.BoundedFloatText(value=4.,min=.001,max=60.,description='Duration (s)')
        self.max_oversampling=widgets.BoundedIntText(value=128,min=1,max=1024,description='Max factor')
        self.density=widgets.FloatText(value=ParameterBox().density,description='Fixed density')
        self.length=widgets.FloatText(value=ParameterBox().length,description='Fixed length')
        self.excitation=widgets.Dropdown(options=[('Eight sine modes','multimode'),('First mode (reference)','first_mode'),('Raised cosine','raised_cosine')],value='multimode',description='Excitation')
        self.target_fields=[];self.initial_fields=[]
        self.optimize_boxes=[widgets.Checkbox(value=True,indent=False,layout=widgets.Layout(width='90px')) for _ in self.names]
        self.target_rows=widgets.VBox([widgets.HTML('Prepare the parameter ranges first.')])
        self.initial_rows=widgets.VBox([widgets.HTML('Prepare the parameter ranges first.')])
        self.grid_output=widgets.Output();self.target_output=widgets.Output();self.initial_output=widgets.Output()
        self.run_outputs={name:widgets.Output() for name in ('adam','lbfgs')}
        self.prediction_outputs={name:widgets.Output() for name in ('adam','lbfgs')}
        self.learning_rates={name:widgets.FloatText(value=lr,description='Learning rate') for name,lr in [('adam',.01),('lbfgs',1.)]}
        self.max_steps={name:widgets.BoundedIntText(value=500,min=1,max=10000,description='Max steps') for name in ('adam','lbfgs')}
        for field in self.configuration_controls():
            if field is not self.coordinates: field.observe(self._configuration_changed,names='value')
        self.coordinates.observe(self._coordinates_changed,names='value')
        self._update_range_rows()
        for field in self.optimize_boxes: field.observe(lambda _:self._clear_results(),names='value')

    def configuration_controls(self):
        return [field for pair in self.range_fields for field in pair]+[self.coordinates,self.sample_rate,self.duration,self.max_oversampling,self.density,self.length,self.excitation,self.device,self.backend]

    def _button(self,label,callback,style=''):
        button=widgets.Button(description=label,button_style=style,layout=widgets.Layout(width='auto'))
        button.on_click(callback);self.buttons.append(button)
        return button

    def _message(self,output,message):
        with output:
            clear_output(wait=False);print(message)

    def _clear_results(self):
        self.results.clear()
        for output in [*self.run_outputs.values(),*self.prediction_outputs.values()]:
            self._message(output,'No current fit. Generate target and initial audio, then run an optimizer.')

    def _configuration_changed(self,_):
        if self.preparing:return
        self.grid=self.renderer=self.target=self.initial=self.criterion=None
        for control in self.target_fields+self.initial_fields: control.disabled=True
        self._clear_results()
        self._message(self.grid_output,'Configuration changed. Apply ranges / prepare grid again.')
        self._message(self.target_output,'No current target.')
        self._message(self.initial_output,'No current initial signal.')

    def _target_changed(self,_):
        if self.preparing:return
        self.target=self.initial=self.criterion=None
        self._clear_results()
        self._message(self.target_output,'Generate the target again. The prepared grid is unchanged.')
        self._message(self.initial_output,'Generate the target and initial signal again.')

    def _initial_changed(self,_):
        if self.preparing:return
        self.initial=None;self._clear_results()
        self._message(self.initial_output,'Generate the initial signal again. The prepared grid is unchanged.')

    def _update_range_rows(self):
        rows=[widgets.HTML('<b>Parameter ranges used for both sounds and the entire fit</b>')]
        rows.append(widgets.HBox([widgets.Label('Parameter',layout=widgets.Layout(width='240px')),widgets.Label('Minimum',layout=widgets.Layout(width='170px')),widgets.Label('Maximum')]))
        for label,unit,pair in zip(self.labels,self.units,self.range_fields):
            rows.append(widgets.HBox([widgets.Label(f'{label} ({unit})',layout=widgets.Layout(width='240px')),*pair]))
        self.range_rows.children=rows
        for field in (self.density,self.length):
            field.layout.display='none' if self.coordinates.value=='material' else ''

    def _coordinates_changed(self,_):
        self._configuration_changed(None)
        box=MaterialParameterBox() if self.coordinates.value=='material' else ParameterBox()
        self.labels=MATERIAL_LABELS if self.coordinates.value=='material' else PERCEPTUAL_LABELS
        self.names,self.units=box.names,box.units
        self.range_fields=[(widgets.FloatText(value=low,layout=widgets.Layout(width='170px')),
                            widgets.FloatText(value=high,layout=widgets.Layout(width='170px'))) for low,high in box.ranges]
        for pair in self.range_fields:
            for field in pair:field.observe(self._configuration_changed,names='value')
        self.optimize_boxes=[widgets.Checkbox(value=True,indent=False,layout=widgets.Layout(width='90px')) for _ in self.names]
        for field in self.optimize_boxes:field.observe(lambda _:self._clear_results(),names='value')
        self.target_fields=[];self.initial_fields=[]
        self.target_rows.children=[widgets.HTML('Coordinate system changed. Prepare the ranges again.')]
        self.initial_rows.children=[widgets.HTML('Prepare the ranges again.')]
        self._update_range_rows()
        self._message(self.grid_output,'Coordinate system changed; ranges reset to reference defaults. Apply ranges / prepare grid.')

    def ranges_panel(self):
        return widgets.VBox([self.coordinates,widgets.HBox([self.device,self.backend]),self.range_rows,widgets.HBox([self.sample_rate,self.duration,self.max_oversampling]),
            widgets.HBox([self.density,self.length]),self.excitation,
            self._button('Apply ranges / prepare grid',lambda _:self.prepare(),'primary'),self.grid_output])

    def _rows(self,fields,mask=False):
        return [widgets.HBox([widgets.Label(label,layout=widgets.Layout(width='240px')),field]+([check] if mask else []))
                for label,field,check in zip(self.labels,fields,self.optimize_boxes)]

    def prepare(self):
        self._configuration_changed(None)
        try:
            ranges=tuple((a.value,b.value) for a,b in self.range_fields)
            box=(MaterialParameterBox(ranges) if self.coordinates.value=='material' else
                 ParameterBox(ranges,self.density.value,self.length.value))
            grid=GridConfig(box,self.sample_rate.value,self.max_oversampling.value)
            renderer=Renderer(grid,self.duration.value,self.excitation.value,device=self.device.value,backend=self.backend.value)
            low,high=box.bounds();defaults=box.defaults().clamp(low,high)
            initial=box.perturb(defaults)
            self.preparing=True
            def sliders(values,callback):
                fields=[]
                for i,(value,(lo,hi)) in enumerate(zip(values.tolist(),box.ranges)):
                    if i<box.positive_count:
                        control=widgets.FloatLogSlider(value=value,base=10,min=math.log10(lo),max=math.log10(hi),step=.001,
                            readout_format='.6g',continuous_update=False,layout=widgets.Layout(width='430px'))
                    else:
                        control=widgets.FloatSlider(value=value,min=lo,max=hi,step=(hi-lo)/1000,
                            readout_format='.6g',continuous_update=False,layout=widgets.Layout(width='430px'))
                    control.observe(callback,names='value');fields.append(control)
                return fields
            self.target_fields=sliders(defaults,self._target_changed)
            self.initial_fields=sliders(initial,self._initial_changed)
            self.target_rows.children=self._rows(self.target_fields,True)
            self.initial_rows.children=self._rows(self.initial_fields)
            self.grid,self.renderer=grid,renderer
            self._message(self.grid_output,grid.describe()+f'\nFull duration: {renderer.duration:g} s; {round(renderer.duration*grid.integration_rate):,} internal steps.\nGrid and rate are now frozen until ranges/settings are reapplied.')
        except Exception as error:
            self._message(self.grid_output,f'Configuration not prepared: {error}')
        finally:self.preparing=False

    def target_panel(self):
        return widgets.VBox([widgets.HTML('Choose target values. Checked parameters are fitted; unchecked parameters keep their initial values.'),self.target_rows,
            self._button('Generate target / synthesis',lambda _:self.generate_target(),'primary'),self.target_output])

    def fill_initial(self,perturb=False):
        if self.grid is None:return
        values=torch.tensor([f.value for f in self.target_fields],dtype=torch.float64)
        if perturb:values=self.grid.box.perturb(values)
        for field,value in zip(self.initial_fields,values.tolist()):field.value=value

    def initial_panel(self):
        return widgets.VBox([self.initial_rows,widgets.HBox([
            self._button('Copy target',lambda _:self.fill_initial()),
            self._button('Signed 10% perturbation',lambda _:self.fill_initial(True)),
            self._button('Generate initial signal',lambda _:self.generate_initial(),'primary')]),self.initial_output])

    def optimizer_panel(self,name):
        return widgets.VBox([widgets.HBox([self.learning_rates[name],self.max_steps[name]]),
            widgets.HTML('One full-duration, full-band CeL loss. Patience: less than 10% best-loss improvement over 20 steps.'),
            self._button(f'Run {name.upper()} from initial',lambda _:self.run(name),'success'),self.run_outputs[name]])

    def prediction_panel(self,name):
        return widgets.VBox([self._button('Show prediction',lambda _:self.show_prediction(name)),self.prediction_outputs[name]])

    def generate_target(self):
        try:
            if self.renderer is None:raise ValueError('Apply ranges / prepare grid first.')
            p=torch.tensor([f.value for f in self.target_fields],dtype=torch.float64)
            audio=self.renderer.synthesize(p)
            self.criterion=criterion_for(audio,self.grid.sample_rate) if audio.numel()>=1024 else None
            settings=dict(sample_rate=self.grid.sample_rate,duration=self.renderer.duration,
                          intervals=self.grid.intervals,oversampling=self.grid.oversampling)
            self.target=dict(physical=p,audio=audio,settings=settings);self.initial=None
            self._clear_results();self._refresh_audio()
        except Exception as error:self._message(self.target_output,f'Target not generated: {error}')

    def generate_initial(self):
        try:
            if self.target is None:raise ValueError('Generate the target first.')
            if self.criterion is None:raise ValueError('Fitting needs at least 1024 output samples; increase duration and prepare again.')
            p=torch.tensor([f.value for f in self.initial_fields],dtype=torch.float64)
            audio=self.renderer.synthesize(p)
            self.initial=dict(physical=p,audio=audio,loss=self.criterion(audio).item())
            self._clear_results();self._refresh_audio()
        except Exception as error:self._message(self.initial_output,f'Initial signal not generated: {error}')

    def _gain(self):
        signals=[self.target['audio']]
        if self.initial is not None:signals.append(self.initial['audio'])
        signals += [r['audio'] for r in self.results.values()]
        return .9/max(max(s.abs().max().item() for s in signals),1e-30)

    def _refresh_audio(self):
        for name,signal,output in [('Target',self.target,self.target_output),('Initial',self.initial,self.initial_output)]:
            if signal is None:continue
            with output:
                clear_output(wait=True)
                p=signal['physical'];physical=self.grid.box.to_material(p)
                derived=perceptual_from_material(physical)
                print(f"{name}: derived f₀={derived[0]:.6g} Hz; linear f₁={derived[0]*torch.sqrt(1+derived[1]):.6g} Hz; T={physical[1]:.6g} N; E={physical[3]:.6g} Pa")
                print(f'First-mode nonlinear-strength indicator B(a/r)²={nonlinear_strength(derived):.6g}; this is not an accuracy bound for the chosen excitation.')
                print(f'Current stability utilisation: {self.grid.score_values(p):.6f}. '+('CeL: '+str(signal['loss']) if name=='Initial' else ''))
                display(Audio((signal['audio']*self._gain()).detach().cpu().numpy(),rate=self.grid.sample_rate,normalize=False))
                rows=modal_diagnostics(p,self.grid)
                def fmt(value):return 'nonoscillatory' if value is None else f'{value:.5g}'
                table=''.join('<tr>'+''.join(f'<td style="padding:3px 12px">{v}</td>' for v in [row['mode'],fmt(row['continuous_hz']),fmt(row['discrete_hz']),fmt(row['cents'])])+'</tr>' for row in rows)
                display(HTML('<table><tr><th>Mode</th><th>Continuous damped Hz</th><th>Discrete damped Hz</th><th>Error (cents)</th></tr>'+table+'</table>'))
                if name=='Target':
                    audio=signal['audio'].detach().cpu();fig,axes=plt.subplots(1,2,figsize=(12,3),layout='constrained')
                    axes[0].plot(torch.arange(audio.numel())/self.grid.sample_rate,audio,linewidth=.6)
                    axes[0].set(xlabel='Time (s)',ylabel='Displacement (m)')
                    spectrum=torch.fft.rfft(audio*torch.hann_window(audio.numel(),dtype=torch.float64)).abs()
                    axes[1].plot(torch.fft.rfftfreq(audio.numel(),1/self.grid.sample_rate),20*torch.log10((spectrum/spectrum.max().clamp_min(1e-30)).clamp_min(1e-6)))
                    axes[1].set(xlabel='Frequency (Hz)',ylabel='Magnitude (dB)',xlim=(0,min(3000,self.grid.sample_rate/2)),ylim=(-100,3))
                    display(fig);plt.close(fig)

    def run(self,name):
        if self.busy:return
        output=self.run_outputs[name]
        controls=self.configuration_controls()+self.target_fields+self.initial_fields+self.optimize_boxes+list(self.learning_rates.values())+list(self.max_steps.values())+self.buttons
        previous_disabled=[control.disabled for control in controls]
        try:
            if self.target is None or self.initial is None:raise ValueError('Generate target and initial signals first.')
            self.busy=True
            for control in controls:control.disabled=True
            with output:
                clear_output(wait=True)
                def progress(row):
                    if row['iteration']%10==0:print(f"Step {row['iteration']} | best CeL {row['best_loss']:.6g} | {row['evaluations']} evaluations",flush=True)
                result=fit(self.renderer,self.target['audio'],self.initial['physical'],[c.value for c in self.optimize_boxes],
                           name,self.learning_rates[name].value,self.max_steps[name].value,progress,self.criterion)
                result['audio']=self.renderer.synthesize(result['best_physical'])
                result['verified_loss']=self.criterion(result['audio']).item()
                if not math.isclose(result['verified_loss'],result['best_loss'],rel_tol=1e-6,abs_tol=1e-10):raise RuntimeError('Best checkpoint verification failed.')
                self.results[name]=result
                path=Path('output')/f'notebook_{self.grid.box.coordinate_system}_{name}.json'
                save_result(path,result,self.renderer,self.target['physical'])
                print(f"{result['iterations']} steps; {result['seconds']:.1f} s. {result['stop_reason']}\nHistory saved to {path.resolve()}")
                if not torch.equal(result['effective_initial'],result['requested_initial']):
                    print('Finite bounded coordinates may nudge selected endpoint starts inward; exact effective start is recorded in the result.')
            self._refresh_audio()
            for key in self.results:self.show_prediction(key)
        except Exception as error:
            self._message(output,f'Fit not completed: {error}')
        finally:
            self.busy=False
            for control,disabled in zip(controls,previous_disabled):control.disabled=disabled

    def show_prediction(self,name):
        output=self.prediction_outputs[name]
        with output:
            clear_output(wait=True)
            if name not in self.results:
                print('Run this optimizer first. Predictions update automatically when it finishes.')
                return
            result=self.results[name]
            print(f"{name.upper()} best CeL: {result['best_loss']:.8g}. {result['stop_reason']}")
            display(Audio((result['audio']*self._gain()).detach().cpu().numpy(),rate=self.target['settings']['sample_rate'],normalize=False))
            rows=''.join('<tr>'+''.join(f'<td style="padding:4px 12px">{value}</td>' for value in
                [html.escape(self.labels[i]),f'{self.target["physical"][i]:.7g}',f'{self.initial["physical"][i]:.7g}',
                 f'{result["best_physical"][i]:.7g}','Yes' if result['mask'][i] else 'Fixed'])+'</tr>' for i in range(len(self.names)))
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
                artist=axis.pcolormesh(times.cpu().numpy(),frequency.cpu().numpy(),db.cpu().numpy(),shading='auto',vmin=-80,vmax=0,cmap='magma',rasterized=True)
                axis.set(title=title,xlabel='Time (s)',ylim=(0,min(2000,self.target['settings']['sample_rate']/2)))
            axes[0].set_ylabel('Frequency (Hz)')
            fig.colorbar(artist,ax=axes,label='Power (dB, shared reference)')
            display(fig);plt.close(fig)
