import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common import RUN, batches, read
from parallel_exp2 import selected_tasks
import supervise


class ParallelExp2Tests(unittest.TestCase):
    def test_exp2_partition_preserves_all_fixed_inputs_once(self):
        identities = []
        for stage, offset, keys in batches():
            tasks = read(RUN / 'inputs' / f'{stage}_{offset:04d}' / 'prepared_tasks.json')
            selected = selected_tasks(tasks)
            self.assertEqual(len(selected), len(keys))
            self.assertEqual([t['key'] for t in selected], keys)
            self.assertTrue(all(t['setting'] == 'experiment2' for t in selected))
            identities.extend(t['key'] for t in selected)
        self.assertEqual(len(identities), 4000)
        self.assertEqual(len(set(identities)), 4000)

    def test_reservation_requires_live_worker_and_running_status(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'parallel_exp2_reservation.json'
            with patch.object(supervise, 'RUN', Path(temp)):
                self.assertEqual(supervise.external_reservations(), [])
                path.write_text(json.dumps({'status': 'RUNNING', 'pid': os.getpid(), 'memory_fraction': .28}))
                self.assertEqual(len(supervise.external_reservations()), 1)
                path.write_text(json.dumps({'status': 'RELEASED', 'pid': os.getpid(), 'memory_fraction': 0}))
                self.assertEqual(supervise.external_reservations(), [])
                path.write_text(json.dumps({'status': 'RUNNING', 'pid': 999999999, 'memory_fraction': .28}))
                self.assertEqual(supervise.external_reservations(), [])


if __name__ == '__main__':
    unittest.main()
