import json
import unittest
from types import SimpleNamespace

from common import protocol
from compatibility import parse_task, update_record


class CompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = protocol.modules()
        cls.task = {'setting': 'experiment1'}

    def parse(self, value):
        return parse_task(self.task, json.dumps(value), self.modules)

    def test_misspelling_restores_multiple_answers_without_gt(self):
        value = {'selected_objectnames': ['cargo1', 'cargo2'],
                 'selected_objects': [{'object_name': 'cargo1'}, {'object_name': 'cargo2'}]}
        ok, parsed, extra = self.parse(value)
        self.assertTrue(ok)
        self.assertEqual(parsed['selected_object_names'], ['cargo1', 'cargo2'])
        self.assertEqual(parsed['selected_objectnames'], value['selected_objectnames'])
        self.assertTrue(extra['compatibility_applied'])

    def test_existing_evaluator_fallback_and_abstention(self):
        for value, expected in [({'selected_objects': [{'object_name': 'door'}]}, ['door']),
                                ({'selected_object_name': 'switch'}, ['switch']),
                                ({'selected_objectnames': []}, []), ({'selected_objects': []}, [])]:
            ok, parsed, _ = self.parse(value)
            self.assertTrue(ok)
            self.assertEqual(parsed['selected_object_names'], expected)

    def test_explicit_canonical_answer_wins_and_malformed_output_is_not_guessed(self):
        ok, parsed, extra = self.parse({'selected_object_names': [], 'selected_objectnames': ['door']})
        self.assertTrue(ok)
        self.assertEqual(parsed['selected_object_names'], [])
        self.assertFalse(extra['compatibility_applied'])
        for value in [{'selected_object_names': 'door', 'selected_object_name': 'door'},
                      {'reasoning_summary': 'door'}, {'referent_mentions': [{'matched_object_names': ['door']}]}]:
            self.assertFalse(self.parse(value)[0])
        self.assertFalse(parse_task(self.task, '{"selected_objectnames":[', self.modules)[0])

    def test_record_keeps_raw_and_generation_hash_and_is_idempotent(self):
        raw = '{"selected_objectnames":["door"]}'
        record = {'model_raw_output': raw, 'request_sha256': 'unchanged', 'parse_ok': False}
        updated = update_record(self.task, record, self.modules)
        self.assertEqual(updated['model_raw_output'], raw)
        self.assertEqual(updated['request_sha256'], 'unchanged')
        self.assertFalse(updated['strict_parse_snapshot']['parse_ok'])
        again = update_record(self.task, json.loads(json.dumps(updated)), self.modules)
        self.assertEqual(updated, again)

    def test_experiment2_uses_original_parser(self):
        task = {'setting': 'experiment2', 'panel_meta': [], 'coordinate_policy': 'paired_canvas_map'}
        raw = '{"referents":[]}'
        self.assertEqual(parse_task(task, raw, self.modules), protocol.parse_task(task, raw, self.modules))


if __name__ == '__main__':
    unittest.main()
