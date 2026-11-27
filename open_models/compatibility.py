"""Deterministic experiment-1 schema compatibility; never use GT or infer labels."""
import copy

from common import protocol

VERSION = 'anchor_schema_compat_v1'


def parse_task(task, raw, modules):
    if task['setting'] != 'experiment1':
        return protocol.parse_task(task, raw, modules)
    payload, warnings = modules.anchor.parse_3d_response_text(raw)
    extra = {'validation_warnings': warnings, 'parser_version': VERSION,
             'compatibility_applied': False, 'compatibility_source': None}
    if not isinstance(payload, dict):
        return False, {}, extra
    parsed = copy.deepcopy(payload)
    if 'selected_object_names' in parsed:
        # An explicitly malformed canonical field is not overwritten by another answer.
        ok = isinstance(parsed['selected_object_names'], list)
        return ok, parsed if ok else {}, extra
    if isinstance(parsed.get('selected_objectnames'), list) and all(
            isinstance(name, str) for name in parsed['selected_objectnames']):
        parsed['selected_object_names'] = list(parsed['selected_objectnames'])
        extra.update(compatibility_applied=True, compatibility_source='selected_objectnames')
        return True, parsed, extra
    # Use exactly the answer containers already supported by the original evaluator.
    # An explicit empty list is a valid abstention, distinct from missing answer fields.
    if isinstance(parsed.get('selected_objects'), list):
        objects = parsed['selected_objects']
        if all(isinstance(item, str) or (isinstance(item, dict) and
               isinstance(item.get('object_name', item.get('selected_object_name')), str)) for item in objects):
            parsed['selected_object_names'] = modules.evaluator1.find_prediction_labels({'parsed_response': payload})
            extra.update(compatibility_applied=True, compatibility_source='original_evaluator:selected_objects')
            return True, parsed, extra
    if isinstance(parsed.get('selected_object_name'), str):
        parsed['selected_object_names'] = modules.evaluator1.find_prediction_labels({'parsed_response': payload})
        extra.update(compatibility_applied=True, compatibility_source='original_evaluator:selected_object_name')
        return True, parsed, extra
    return False, {}, extra


def update_record(task, record, modules):
    """Keep generation identity and raw bytes; record the original strict outcome."""
    if task['setting'] != 'experiment1':
        return record
    ok, parsed, extra = parse_task(task, record['model_raw_output'], modules)
    if 'strict_parse_snapshot' not in record:
        strict_ok, _, strict_extra = protocol.parse_task(task, record['model_raw_output'], modules)
        record['strict_parse_snapshot'] = {
            'parse_ok': strict_ok, 'response_status': 'ok' if strict_ok else 'invalid',
            'validation_warnings': strict_extra.get('validation_warnings', [])}
    record.update(parse_ok=ok, parsed_json=parsed, parsed_response=parsed,
                  response_status='ok' if ok else 'invalid', **extra)
    return record
