"""Declared full-clause output schema for the bounded correspondence stage.

No captions, selection decisions, target images, or validator rules change.
The original single-anchor example caused the model to omit change verdicts.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import auto_extraction as ae, qwen35_state_delta_resume as recovery
from .artifacts import sha256, write_json


def complete_clause_schema(prompt):
    marker = 'INPUT METADATA (not the output schema):\n'
    if not prompt.startswith(marker):
        raise ValueError('correspondence metadata missing')
    metadata, _ = json.JSONDecoder().raw_decode(prompt[len(marker):])
    clauses = metadata['clauses']
    ids = [row['id'] for row in clauses]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('invalid supplied clause IDs')
    start = prompt.index('Return ONLY JSON:')
    end = prompt.index('Do not convert uncertainty', start)
    schema = dict(verdicts=[dict(id=key, match='target|neighbor|neither|uncertain',
                                evidence_frames=[metadata['target_ids'][0]]) for key in ids])
    replacement = ('Return ONLY JSON with exactly ' + str(len(ids)) + ' verdicts, one per supplied clause.\n'
        + json.dumps(schema) + '\n'
        'The match string lists alternatives: choose exactly ONE category for each clause after inspecting the images. '
        'Replace example evidence with the actual supporting SOURCE IDs. '
        'Do not omit a clause. Required IDs in order: ' + json.dumps(ids) + '.\n')
    return prompt[:start] + replacement + prompt[end:]


def declare(root):
    path = root / 'correspondence_schema_extension.json'
    value = dict(policy='complete_clause_schema_v1', module=str(Path(__file__).resolve()),
        module_sha256=sha256(__file__), original_plan_signature=ae.read(root/'plan.json')['signature'],
        change='Replace single-anchor schema example with every actual clause ID and category alternatives',
        previous_failed_responses_retained=True, same_targets=True, same_images=True,
        same_validator=True, same_removal_only_policy=True, reference_access=False)
    if path.exists() and ae.read(path) != value:
        raise ValueError('correspondence schema extension changed')
    if not path.exists():
        write_json(path, value)


class CompleteClauseSession(recovery.RecoverySession):
    def call(self, role, ids, prompt, validator, tokens=640):
        if role == 'correspondence':
            declare(self.root)
            prompt = complete_clause_schema(prompt)
        return super().call(role, ids, prompt, validator, tokens)


def main():
    recovery.RecoverySession = CompleteClauseSession
    recovery.main()


if __name__ == '__main__':
    main()
