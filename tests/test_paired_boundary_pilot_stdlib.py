"""Synthetic lifecycle checks; no torch, policy, dataset or physics imports."""
import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('boundary_pilot', ROOT/'scripts/paired_boundary_pilot.py')
pilot=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class BoundaryPilotTest(unittest.TestCase):
    def test_comparison_has_no_broadcast_or_tolerance(self):
        self.assertFalse(pilot.compare_arrays([1], [1, 1])['exact'])
        self.assertFalse(pilot.compare_arrays([1], [1+1e-12])['exact'])
        self.assertTrue(pilot.compare_arrays([1, 2], [1, 2])['exact'])

    def test_comparison_rejects_nonfinite(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                pilot.compare_arrays([value], [value])

    def test_source_only_import(self):
        # Explicit import of the module above does not construct optional runtimes.
        self.assertNotIn('torch', pilot.__dict__)
        self.assertNotIn('mujoco', pilot.__dict__)

    def test_absolute_and_parent_paths_rejected(self):
        for name in ('../outside', '/outside', 'C:/outside', 'a\\outside'):
            with self.assertRaises(ValueError):
                pilot.resolve(ROOT, name)

    def test_resolved_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as other:
            root=Path(d); outside=Path(other)/'file';outside.write_text('x')
            (root/'link').symlink_to(outside)
            with self.assertRaises(ValueError):
                pilot.resolve(root, 'link')

    def test_actual_calls_logged_once_and_restored(self):
        class Step:
            def __init__(self):self.calls=0
            def __call__(self, value):
                self.calls+=1
                return {'observation.state':value['observation.state']+1}
        step=Step(); original=Step.__call__
        class Pipeline:steps=[step]
        with pilot.observe_processor_states(Pipeline(),copy.deepcopy) as rows:
            self.assertEqual(step({'observation.state':2}),{'observation.state':3})
        self.assertEqual(step.calls,1)
        self.assertEqual(rows,[{'index':0,'before':2,'after':3}])
        self.assertIs(Step.__call__,original)

    def test_unrelated_same_class_step_not_observed(self):
        class Step:
            def __call__(self,value):return value
        a,b=Step(),Step()
        class Pipeline:steps=[a]
        with pilot.observe_processor_states(Pipeline(),copy.deepcopy) as rows:
            b({'observation.state':2})
            a({'observation.state':3})
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['before'],3)

    def test_failure_restores_method(self):
        class Step:
            def __call__(self,value):raise ValueError('test-failure')
        step=Step(); original=Step.__call__
        class Pipeline:steps=[step]
        with self.assertRaises(ValueError):
            with pilot.observe_processor_states(Pipeline(),copy.deepcopy):step({})
        self.assertIs(Step.__call__,original)

    def test_later_owner_not_overwritten(self):
        class Step:
            def __call__(self,value):return value
        original=Step.__call__;step=Step()
        class Pipeline:steps=[step]
        later=lambda self,value:value
        try:
            with self.assertRaises(RuntimeError):
                with pilot.observe_processor_states(Pipeline(),copy.deepcopy):Step.__call__=later
            self.assertIs(Step.__call__,later)
        finally:Step.__call__=original

    def test_inherited_method_restored_without_new_override(self):
        class Parent:
            def __call__(self,value):return value
        class Child(Parent):pass
        step=Child()
        class Pipeline:steps=[step]
        with pilot.observe_processor_states(Pipeline(),copy.deepcopy):step({})
        self.assertNotIn('__call__',Child.__dict__)

    def test_aliased_processor_rejected(self):
        class Step:
            def __call__(self,value):return value
        step=Step()
        class Pipeline:steps=[step,step]
        with self.assertRaises(ValueError):
            with pilot.observe_processor_states(Pipeline(),copy.deepcopy):pass

    def test_mixed_inheritance_does_not_double_capture(self):
        class Parent:
            def __call__(self,value):return value
        class Child(Parent):pass
        a,b=Parent(),Child()
        class Pipeline:steps=[a,b]
        with pilot.observe_processor_states(Pipeline(),copy.deepcopy) as rows:
            a({'observation.state':1})
            b({'observation.state':2})
        self.assertEqual([row['index'] for row in rows],[0,1])

    def test_execution_refuses_missing_admission_before_ml_imports(self):
        # Replace only source-validation for this synthetic missing-receipt check.
        original=pilot.validate
        try:
            pilot.validate=lambda value:None
            with self.assertRaisesRegex(ValueError,'admission'):
                pilot.execute({},Path('not-created'),Path('not-used'))
        finally:pilot.validate=original


if __name__=='__main__':unittest.main()
