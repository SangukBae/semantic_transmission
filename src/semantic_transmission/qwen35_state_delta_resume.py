"""Audited JSON nesting normalization; no caption or evidence values are edited.

The failed Qwen source response remains verbatim. A declared parser extension
closes the anchor object or moves known top-level keys out of that object.
The old plan, successful requests and delivered Astra packets remain unchanged.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import re
import time

from . import auto_extraction as ae, qwen35_state_delta as sd
from .artifacts import sha256, write_json

ORIGINAL_PARSE = ae.parse_json


def semantic_fields(raw):
    decoder = json.JSONDecoder()
    fields = []
    pattern = r'(?<!\\)"(text|evidence_frames|before|after|uncertainties|needs_check)"\s*:\s*'
    for match in re.finditer(pattern, raw):
        value, _ = decoder.raw_decode(raw[match.end():])
        fields.append((match.group(1), value))
    if not fields or not any(k == 'text' for k, v in fields):
        raise ValueError('no complete caption fields available for syntax normalization')
    return fields


def normalized_json(raw):
    mode = None
    try:
        value = ORIGINAL_PARSE(raw)
    except (ValueError, TypeError):
        # Only this observed missing-anchor-brace pattern is recoverable. Never
        # add text, source IDs, changes, uncertainty values, or boolean defaults.
        match = re.search(r'(?<!\\),\s*"changes"\s*:', raw)
        if not match:
            raise
        value = ORIGINAL_PARSE(raw[:match.start()] + '}' + raw[match.start():])
        mode = 'close_anchor_before_changes'
    if (set(value) == {'anchor'} and isinstance(value['anchor'], dict)
            and set(value['anchor']) == {'text', 'evidence_frames', 'changes', 'uncertainties', 'needs_check'}):
        anchor = value['anchor']
        value = dict(anchor=dict(text=anchor['text'], evidence_frames=anchor['evidence_frames']),
                     changes=anchor['changes'], uncertainties=anchor['uncertainties'], needs_check=anchor['needs_check'])
        mode = 'move_existing_top_level_fields_out_of_anchor'
    if mode:
        if (set(value) != {'anchor', 'changes', 'uncertainties', 'needs_check'}
                or semantic_fields(raw) != semantic_fields(json.dumps(value))):
            raise ValueError('normalization changed caption or evidence fields')
    return value, mode


def install_normalizer():
    ae.parse_json = lambda raw: normalized_json(raw)[0]


def declare(root):
    path = root / 'syntax_recovery_extension_v3.json'
    value = dict(policy='json_nesting_only_all_caption_fields_unchanged_v3',
        module=str(Path(__file__).resolve()), module_sha256=sha256(__file__),
        original_plan_signature=ae.read(root / 'plan.json')['signature'],
        allowed_changes=['close anchor brace immediately before changes', 'move existing changes/uncertainties/needs_check from anchor to top level'],
        semantic_field_equality_required=True, defaults_or_new_values_forbidden=True,
        additional_model=False, additional_inference_for_normalization=False,
        reference_access=False, delivered_evaluation_packets_modified=False,
        earlier_failed_formatter_attempts_retained=True)
    if path.exists() and ae.read(path) != value:
        raise ValueError('normalization extension changed')
    if not path.exists():
        write_json(path, value)
    return path


class RecoverySession(sd.Session):
    def trace(self, raw, parent=None):
        _, mode = normalized_json(raw)
        extension = declare(self.root)
        trace = dict(mode=mode, extension_path=str(extension), extension_sha256=sha256(extension),
                     preserved_fields=semantic_fields(raw), raw_response_unchanged=True)
        if parent:
            trace.update(parent_path=str(parent), parent_sha256=sha256(parent))
        return trace

    def validate_record(self, record, validator):
        super().validate_record(record, validator)
        _, mode = normalized_json(record['raw_text'])
        if mode and 'syntax_normalization' not in record:
            record['syntax_normalization'] = self.trace(record['raw_text'])

    def call(self, role, ids, prompt, validator, tokens=640):
        if role == 'state_caption':
            repaired = sd.explicit_source_examples(prompt, ids)
            repaired += '\nVALID SOURCE FRAME IDS: ' + json.dumps(ids) + '. Use these exact IDs, never image positions or example numbers. Use compact JSON without indentation.'
            identity = dict(plan_signature=self.plan['signature'], role=role + '_schema', ids=ids,
                frame_hashes=[self.plan['source_frame_hashes'][str(i)] for i in ids], prompt=repaired,
                labels=[f"SOURCE frame {i}, {i/self.plan['fps']:.6f}s" for i in ids], tokens=tokens)
            key = ae.digest(identity)
            parent = self.root / 'failed_attempts' / identity['role'] / f'{key}_2.json'
            output = self.root / 'calls' / identity['role'] / f'{key}.json'
            if parent.exists() and not output.exists():
                record = ae.unseal(parent)
                value, mode = normalized_json(record['raw_text'])
                if not mode or record['identity'] != identity:
                    raise ValueError('failed response cannot be normalized without changes')
                record['result'] = value
                record['syntax_normalization'] = self.trace(record['raw_text'], parent)
                record['original_validation_error'] = record.pop('error')
                self.validate_record(record, validator)
                write_json(output, ae.seal(record))
                self.calls.append(dict(role=identity['role'], path=str(output), sha256=sha256(output), reused=True))
                print(json.dumps(dict(role='syntax_normalization', ids=ids, additional_inference=False)), flush=True)
                return value
        return super().call(role, ids, prompt, validator, tokens)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'check'))
    parser.add_argument('--root', type=Path, default=sd.ROOT)
    parser.add_argument('--arm', choices=sd.ARMS)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args(argv)
    install_normalizer()
    if args.action != 'run':
        return sd.main(argv)
    root = args.root.resolve()
    with (root / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = sd.check_plan(root)
        declare(root)
        stage = root / 'execution_stages' / ae.now().replace(':', '-')
        for name in ('runtime.json', 'latest_execution.json'):
            if (root / name).exists():
                write_json(stage / name, ae.read(root / name))
        session = RecoverySession(root, plan)
        start, started = time.monotonic(), ae.now()
        try:
            sd.extract(session, args.arm)
        finally:
            write_json(root / 'latest_execution.json', dict(started_utc=started, ended_utc=ae.now(),
                elapsed_seconds=time.monotonic()-start, calls=len(session.calls), arm=args.arm,
                syntax_extension_sha256=sha256(root / 'syntax_recovery_extension_v3.json')))
        print(json.dumps(sd.verify(root, partial=bool(args.arm))))


if __name__ == '__main__':
    main()
