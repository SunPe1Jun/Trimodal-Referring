import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from common import SETTINGS, batches, protocol, write
from report import report_model
from worker import infer


class ExpansionTests(unittest.TestCase):
    def test_selection_covers_all_4000_once_and_never_runs_experiment3(self):
        jobs = batches()
        keys = [key for _, _, group in jobs for key in group]
        self.assertEqual(len(keys), 4000)
        self.assertEqual(len(set(keys)), 4000)
        self.assertEqual(sum(len(group) for stage, _, group in jobs if stage == 'smoke'), 80)
        self.assertEqual(SETTINGS, ('experiment1', 'experiment2'))

    def test_generated_but_unparsed_response_is_reused_and_drift_is_rejected(self):
        task = {'setting': 'experiment1', 'scene': 'scene1', 'row_index': 0, 'images': []}
        lock = {'model': 'test'}
        request_hash = protocol.sha(json.dumps({'lock': lock, 'task': task}, sort_keys=True).encode())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'experiment1/raw/scene1/row_0.json'
            write(path, {'request_sha256': request_hash, 'response_status': 'unparsed', 'model_raw_output': 'saved'})
            adapter = Mock()
            self.assertEqual(infer(task, Path(tmp), adapter, lock)['model_raw_output'], 'saved')
            adapter.generate.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'differs'):
                infer(task, Path(tmp), adapter, {'model': 'changed'})

    def test_aggregation_uses_counts_instead_of_mean_of_batch_f1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, (tp, fp, fn) in enumerate([(1, 0, 0), (1, 9, 9)]):
                write(root / f'batches/{i}/summary.json', {'requests': 2, 'valid_outputs': 2,
                    'metrics': {'experiment1': {'set_true_positive_count': tp,
                    'set_false_positive_count': fp, 'set_false_negative_count': fn,
                    'exact_match_count_overall': 1 if i == 0 else 0},
                    'experiment2': {'point_tp_100': tp, 'point_fp_100': fp, 'point_fn_100': fn},
                    'experiment2_declared_space': {}}})
            result = report_model(root)
            self.assertAlmostEqual(result['metrics']['experiment1']['micro_f1'], 4 / 22)
            self.assertAlmostEqual(result['metrics']['experiment2']['point_100_f1'], 4 / 22)
            self.assertEqual(result['metrics']['experiment1']['exact'], .5)


if __name__ == '__main__':
    unittest.main()
