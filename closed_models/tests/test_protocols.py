import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from protocols import apply_protocols
from temporal_evidence import choose_coverage


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.task={'setting':'experiment3','instruction':'Inspect the unique purple chair.',
                   'prompt_text':'legacy prompt','images':['image.jpg'],
                   'evidence_json':[{'panel_id':'P1','video_time_seconds':1.,
                     'relative_sample_time_seconds':.5,
                     'telemetry':{'camera_position':[0,0,0], 'gaze_hit':[1,2,3],
                                  'gaze_valid':True,'hand_hit':[4,5,6],'hand_valid':True,
                                  'gaze_to_hand_hit_distance_world_units':3.}}]}

    def test_instruction_ablation_keeps_same_images_and_cues(self):
        before=copy.deepcopy(self.task)
        full=apply_protocols([self.task],'evidence','full')[0]
        removed=apply_protocols([self.task],'evidence','no_instruction')[0]
        self.assertIn(self.task['instruction'],full['prompt_text'])
        self.assertNotIn(self.task['instruction'],removed['prompt_text'])
        self.assertEqual(full['images'],removed['images'])
        self.assertEqual(full['evidence_json'],removed['evidence_json'])
        self.assertEqual(self.task,before)

    def test_gaze_and_hand_occlusion_remove_data_and_derived_distances(self):
        no_gaze=apply_protocols([self.task],'evidence','no_gaze')[0]['prompt_text']
        self.assertNotIn('"gaze_hit"',no_gaze)
        self.assertIn('"hand_hit"',no_gaze)
        self.assertNotIn('gaze_to_hand_hit_distance',no_gaze)
        no_hand=apply_protocols([self.task],'evidence','no_hand')[0]['prompt_text']
        self.assertNotIn('"hand_hit"',no_hand)
        self.assertIn('"gaze_hit"',no_hand)

    def test_visual_ablation_sends_zero_images(self):
        task=apply_protocols([self.task],'evidence','no_visual')[0]
        self.assertEqual(task['images'],[])
        self.assertIn('Images are withheld',task['prompt_text'])

    def test_coverage_reserves_late_evidence_despite_early_score_bias(self):
        candidates=[{'video_time':float(t)+.5,'score':100 if t<2 else 1} for t in range(8)]
        chosen=choose_coverage(candidates,0,8,4)
        self.assertEqual(len(chosen),4)
        self.assertTrue(any(x['video_time']>=6 for x in chosen))
        self.assertEqual(len(set(x['video_time'] for x in chosen)),4)

    def test_invalid_budget_rejected(self):
        with self.assertRaises(ValueError):choose_coverage([],0,1,0)


if __name__=='__main__':unittest.main()
