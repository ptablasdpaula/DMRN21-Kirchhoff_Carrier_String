"""Check both standalone explorers, their widget callbacks, and fresh-kernel execution."""
import contextlib
import io
import math
from pathlib import Path
import unittest

import nbformat
from nbclient import NotebookClient

project_root = Path(__file__).resolve().parents[1]
notebook_names = ('stability_ranges.ipynb', 'oversampling_ranges.ipynb')


def load_definitions(name):
    notebook = nbformat.read(project_root / 'notebooks' / name, as_version=4)
    namespace = {'__name__': '__explorer_test__'}
    for cell in notebook.cells:
        if cell.cell_type == 'code' and 'definitions' in cell.metadata.get('tags', []):
            exec(compile(cell.source, name, 'exec'), namespace)
    return namespace


class StabilityRangeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.explorers = [load_definitions(name) for name in notebook_names]

    def test_scores_factors_and_cross_notebook_agreement(self):
        for namespace in self.explorers:
            parameters = namespace['default_parameters']
            for intervals, score in ((64, 0.07503150021249279),
                                     (128, 0.5945588968817995), (256, 7.089161924036451)):
                actual = namespace['evaluate'](parameters, intervals)
                self.assertAlmostEqual(actual['score'], score, places=12)
            for intervals, expected in ((128, 1), (256, 4), (512, 16)):
                result = namespace['evaluate'](parameters, intervals, adaptive=True)
                self.assertEqual(result['factor'], expected)
                self.assertLessEqual(result['score'], 1)
                if expected > 1:
                    previous = sum(namespace['score_parts'](result['coeff'], 44100 * expected / 2).values())
                    self.assertGreater(previous, 1)
            for intervals in (64, 128, 256, 512):
                fixed = namespace['evaluate'](parameters, intervals)
                adaptive = namespace['evaluate'](parameters, intervals, adaptive=True)
                self.assertEqual(fixed['score'], adaptive['base_score'])
                self.assertEqual(fixed['coeff'], adaptive['coeff'])
        for intervals in (64, 128, 256, 512):
            for adaptive in (False, True):
                results = [ns['evaluate'](ns['default_parameters'], intervals, adaptive) for ns in self.explorers]
                self.assertEqual(results[0], results[1])

    def test_conditional_boundaries_and_invalid_domains(self):
        for namespace in self.explorers:
            parameters = namespace['default_parameters'].copy()
            evaluate = namespace['evaluate']
            result = evaluate(parameters)
            bounds = namespace['conditional_ranges'](parameters, result, 128)
            for key in ('tension', 'modulus', 'sigma_1'):
                limit = bounds[key][1]
                self.assertAlmostEqual(evaluate(parameters | {key: limit})['score'], 1, places=12)
                self.assertTrue(evaluate(parameters | {key: limit * 0.999})['passes'])
                self.assertFalse(evaluate(parameters | {key: limit * 1.001})['passes'])
            for key in ('density', 'length'):
                limit = bounds[key][0]
                self.assertAlmostEqual(evaluate(parameters | {key: limit})['score'], 1, places=12)
                self.assertTrue(evaluate(parameters | {key: limit * 1.001})['passes'])
                self.assertFalse(evaluate(parameters | {key: limit * 0.999})['passes'])
            lower, upper, _ = bounds['radius']
            for radius in (lower, upper):
                self.assertAlmostEqual(evaluate(parameters | {'radius': radius})['score'], 1, places=12)
            self.assertFalse(evaluate(parameters | {'radius': lower * 0.99})['passes'])
            self.assertFalse(evaluate(parameters | {'radius': upper * 1.01})['passes'])
            self.assertTrue(evaluate(parameters | {'radius': math.sqrt(lower * upper)})['passes'])
            maximum = result['maximum_intervals']
            self.assertEqual(maximum, 148)
            self.assertTrue(evaluate(parameters, maximum)['passes'])
            self.assertFalse(evaluate(parameters, maximum + 2)['passes'])

            impossible = parameters | {'tension': 1e6, 'modulus': 1e15}
            impossible_bounds = namespace['conditional_ranges'](impossible, evaluate(impossible), 128)
            self.assertIsNone(impossible_bounds['radius'])
            excessive_loss = parameters | {'sigma_1': result['coeff']['H']**2 / (4 * result['time_step']) * 1.1}
            loss_bounds = namespace['conditional_ranges'](excessive_loss, evaluate(excessive_loss), 128)
            self.assertIsNone(loss_bounds['density'])
            self.assertIsNone(loss_bounds['radius'])
            zeros = parameters | {'tension': 0., 'sigma_0': 0., 'sigma_1': 0.}
            self.assertTrue(evaluate(zeros)['passes'])
            zero_bounds = namespace['conditional_ranges'](zeros, evaluate(zeros), 128)
            self.assertEqual(zero_bounds['radius'][0], 0.)
            self.assertTrue(zero_bounds['radius'][2])
            for key, value in (('density', 0), ('modulus', -1), ('sigma_0', -1),
                               ('radius', float('nan')), ('radius', 1e-300), ('length', float('inf'))):
                self.assertFalse(evaluate(parameters | {key: value})['valid'])
            for intervals in (0, 3, 5):
                self.assertFalse(evaluate(parameters, intervals)['valid'])

    def test_joint_plot_matches_scalar_calculations(self):
        for namespace in self.explorers:
            parameters = namespace['default_parameters']
            for adaptive in (False, True):
                for x_key, y_key, x_limits, y_limits in (
                    ('tension', 'modulus', (0., 400.), (0., 500.)),
                    ('radius', 'length', (0.05, 1.), (0.1, 2.)),
                ):
                    X, Y, values, scores = namespace['region_values'](
                        parameters, 128, x_key, y_key, x_limits, y_limits, adaptive, points=7,
                    )
                    for row in range(7):
                        for column in range(7):
                            current = parameters | {
                                x_key: X[row, column] * namespace['parameter_info'][x_key][2],
                                y_key: Y[row, column] * namespace['parameter_info'][y_key][2],
                            }
                            scalar = namespace['evaluate'](current, 128, adaptive)
                            if not scalar['valid']:
                                self.assertTrue(math.isnan(values[row, column]))
                            else:
                                expected = math.log2(scalar['factor']) if adaptive else int(scalar['passes'])
                                self.assertEqual(values[row, column], expected)
                                self.assertAlmostEqual(scores[row, column], scalar['base_score'], delta=1e-10 * max(1, scalar['base_score']))

    def test_live_callbacks_preserve_bad_inputs_and_refresh(self):
        for adaptive, namespace in zip((False, True), self.explorers):
            original_display = namespace['display']
            namespace['display'] = lambda *args, **kwargs: None
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    explorer = namespace['StabilityExplorer'](adaptive=adaptive)
                    explorer.intervals.value = 256
                    self.assertEqual(explorer.last_result['factor'], 4 if adaptive else 1)
                    self.assertEqual(explorer.last_result['passes'], adaptive)
                    explorer.controls['density'].value = -10
                    self.assertEqual(explorer.controls['density'].value, -10)
                    self.assertFalse(explorer.last_result['valid'])
                    self.assertIn('strictly positive', explorer.report.value)
                    explorer.reset()
                    explorer.controls['tension'].value = 1e7
                    self.assertEqual(explorer.controls['tension'].value, 1e7)
                    self.assertEqual(explorer.last_result['passes'], adaptive)
                    explorer.reset()
                    explorer.x_axis.value = 'radius'
                    self.assertIsNone(explorer.last_plot_error)
                    explorer.y_axis.value = 'radius'
                    self.assertIn('different parameters', explorer.last_plot_error)
                    explorer.reset()
                    explorer.x_min.value = explorer.x_max.value
                    self.assertIn('minimum', explorer.last_plot_error)
                    explorer.reset()
                    self.assertTrue(explorer.last_result['passes'])
                    self.assertEqual(explorer.intervals.value, 128)
                    explorer.widget.close()
            finally:
                namespace['display'] = original_display

    def test_notebooks_execute_in_fresh_kernels(self):
        output_directory = project_root / '.notebook-build'
        output_directory.mkdir(exist_ok=True)
        for name in notebook_names:
            with self.subTest(notebook=name):
                notebook = nbformat.read(project_root / 'notebooks' / name, as_version=4)
                client = NotebookClient(notebook, timeout=120, kernel_name='python3',
                                        resources={'metadata': {'path': str(project_root)}})
                executed = client.execute()
                nbformat.validate(executed)
                for cell in executed.cells:
                    if cell.cell_type == 'code':
                        self.assertIsNotNone(cell.execution_count)
                        self.assertFalse(any(output.output_type == 'error' for output in cell.outputs))
                nbformat.write(executed, output_directory / name.replace('.ipynb', '.executed.ipynb'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
