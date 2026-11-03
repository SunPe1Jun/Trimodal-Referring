"""Explicit experimental protocols; never relabel revised prompts as legacy results."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def evidence_prompt(task, ablation='full'):
    template = (ROOT / 'closed_models/prompts/exp3_evidence_grounding_v1.md').read_text()
    instruction = task['instruction'] if ablation != 'no_instruction' else '[instruction withheld]'
    observations = []
    for item in task['evidence_json']:
        telemetry = copy.deepcopy(item['telemetry'])
        # Derived distances must not retain a removed cue's information.
        telemetry = {k:v for k,v in telemetry.items() if not k.endswith('_distance_world_units')}
        if ablation == 'no_gaze':
            telemetry = {k:v for k,v in telemetry.items() if not k.startswith('gaze_')}
        if ablation == 'no_hand':
            telemetry = {k:v for k,v in telemetry.items() if not k.startswith('hand_')}
        observations.append({'panel_id':item['panel_id'],
                             'video_time_seconds':item['video_time_seconds'],
                             'telemetry_time_seconds':item['relative_sample_time_seconds'],
                             'camera_and_measured_cues':telemetry,
                             'frame_time_contract':item.get('alignment',{})})
    if ablation == 'no_visual':
        template = template.replace('use the chronological images to bind those mentions to visible evidence',
                                    'the images are withheld in this controlled ablation')
        template += '\nImages are withheld. Do not claim to have seen an image; use only available language and telemetry.\n'
    return template.replace('[INSTRUCTION]', instruction).replace('[OBSERVATIONS]', json.dumps(observations,ensure_ascii=False,indent=2))


def apply_protocols(tasks, mode, ablation, legacy_renderer=None):
    transformed = copy.deepcopy(tasks)
    for task in transformed:
        if task['setting'] != 'experiment3':
            continue
        if mode == 'evidence':
            task['prompt_text'] = evidence_prompt(task, ablation)
        elif ablation != 'full':
            if legacy_renderer is None:
                raise ValueError('Legacy ablation renderer required')
            task['prompt_text'] = legacy_renderer('', {'prompt_text':task['prompt_text']}, ablation)
        if ablation == 'no_visual':
            task['images'] = []
        task.update(exp3_prompt_mode=mode, ablation_variant=ablation)
    return transformed
