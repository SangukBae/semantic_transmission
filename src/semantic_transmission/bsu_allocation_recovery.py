"""Resume frozen allocation runs with bounded, audited caption schema retries.

The original protocol, cache optimization, validators and accepted responses are
unchanged. Only exhausted uncached caption requests receive error-specific
feedback. A separate receipt records this changed retry policy and its code.
"""
import argparse
from functools import partial
from pathlib import Path
import time

from . import bsu_allocation_fast as fast
from .artifacts import sha256, write_json
from .qwen35_pairs_video import freeze
from .webvid5 import fingerprint

MODULE = 'semantic_transmission.bsu_allocation_recovery'
PROFILE = 'caption_recovery.json'
MAX_RETRIES = 3
POLICY = 'strict_validator_error_feedback_v1'


def prepare(root):
    execution = fast.verify_execution(root)
    paths = [Path(__file__).resolve(), fast.labels.REPO/'scripts/run_bsu_allocation_recovery.sh']
    value = dict(policy=POLICY, parent_execution_signature=execution['signature'],
        parent_protocol_signature=execution['parent_protocol_signature'],
        max_additional_attempts=MAX_RETRIES,
        code={str(path): sha256(path) for path in paths},
        scope='Retry failed captions only; original validators, models, successful caches and labels preserved')
    value['signature'] = fingerprint(value)
    freeze(Path(root)/PROFILE, value)
    return value


def verify(root):
    execution = fast.verify_execution(root)
    value = fast.labels.read(Path(root)/PROFILE)
    if value.get('signature') != fingerprint({k:v for k,v in value.items() if k != 'signature'}):
        raise ValueError('caption recovery receipt signature differs')
    if (value['parent_execution_signature'] != execution['signature'] or
            value['parent_protocol_signature'] != execution['parent_protocol_signature']):
        raise ValueError('caption recovery belongs to another experiment')
    if value['policy'] != POLICY or value['max_additional_attempts'] != MAX_RETRIES:
        raise ValueError('caption recovery retry policy differs')
    expected = {str(Path(__file__).resolve()), str(fast.labels.REPO/'scripts/run_bsu_allocation_recovery.sh')}
    if set(value['code']) != expected:
        raise ValueError('caption recovery code manifest differs')
    for path, digest in value['code'].items():
        if sha256(path) != digest:
            raise ValueError('caption recovery code changed: '+path)
    return value


def feedback_prompt(prompt, error, task, attempt):
    schema = '{"caption":"concise English description", "uncertainties":[]}'
    if task == 'visual_review':
        schema = '{"caption":"concise English description", "removed_claims":[], "uncertainties":[]}'
    return (prompt + f'\nSchema correction attempt {attempt}. The previous response was rejected: {error}.\n'
        'Generate a new answer from the same source images, following all original content constraints. '
        'Return one complete JSON object with exactly these field names and value types: '+schema+'. '
        'Spell "uncertainties" exactly; "uncertains" is invalid. '
        'Each list must contain strings, or be empty. Keep the caption within 72 words. '
        'Do not omit required fields or add markdown or explanations.')


def infer(vlm, path, identity, images, labels, prompt, validator, tokens=384, *, original, profile):
    from . import auto_extraction as ae, qwen35_pairs_comparison as pc
    path = Path(path)
    args = (vlm, path, identity, images, labels, prompt, validator, tokens)
    # Original code still verifies sealed bytes, request identity and model role.
    if path.exists():
        return original(*args)
    full_identity = dict(identity, prompt_sha256=ae.digest(prompt), labels=labels, max_new_tokens=tokens)

    def failures():
        found = []
        for item in sorted((path.parent/'failed_attempts').glob(path.stem+'_*.json')):
            record = ae.unseal(item)
            if record['identity'] != full_identity:
                continue
            pc.verify_record(vlm.config, record)
            runtime = record['model_runtime']
            if (runtime.get('model_id'), runtime.get('revision'), runtime.get('quantization')) != (pc.MODEL_ID, pc.REVISION, 'nf4'):
                raise ValueError('failed caption model provenance changed')
            found.append((item, record))
        return found

    prior = failures()
    if len(prior) < 3:
        try:
            return original(*args)
        except RuntimeError as exc:
            if str(exc) != f'three invalid responses; successful calls are resumable: {path}':
                raise
        prior = failures()
    if not prior:
        raise ValueError('caption recovery requires preserved failed responses')
    failed_path, failed = prior[-1]
    for attempt in range(1, MAX_RETRIES+1):
        actual_prompt = feedback_prompt(prompt, failed['error'], identity['task'], attempt)
        response = vlm.generate(images, labels, actual_prompt, min(1152, tokens+attempt*192), task=identity['task'])
        record = dict(identity=full_identity, actual_prompt=actual_prompt, created_utc=ae.now(),
            recovery=dict(policy=POLICY, signature=profile['signature'], attempt=attempt,
                preceding_failure=dict(path=str(failed_path), sha256=sha256(failed_path)),
                validator_error=failed['error']), **response)
        # Provenance mismatches are fatal, not treated as response format errors.
        pc.verify_record(vlm.config, record)
        runtime = record['model_runtime']
        if (runtime.get('model_id'), runtime.get('revision'), runtime.get('quantization')) != (pc.MODEL_ID, pc.REVISION, 'nf4'):
            raise ValueError('recovery caption model provenance differs')
        try:
            if response['truncated']:
                raise ValueError('generation reached token limit')
            result = validator(ae.parse_json(response['raw_text']))
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            failed = dict(record, error=str(exc))
            failed_path = path.parent/'failed_attempts'/f'{path.stem}_{time.time_ns()}.json'
            write_json(failed_path, ae.seal(failed))
            continue
        write_json(path, ae.seal(dict(record, result=result)))
        print(f'CAPTION_RECOVERED {path} attempt={attempt}', flush=True)
        return result
    raise RuntimeError(f'{MAX_RETRIES} error-specific caption retries exhausted: {path}')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('command')
    parser.add_argument('--output', type=Path, default=fast.labels.DEFAULT)
    parser.add_argument('--stage')
    args, _ = parser.parse_known_args()
    root = args.output.resolve()
    if args.command == 'prepare':
        print(prepare(root)['signature'])
        return
    if args.command not in ('status', 'watch'):
        profile = verify(root)
        if args.command == 'worker' and args.stage == 'caption':
            from . import qwen35_pairs_comparison as pc
            pc.infer = partial(infer, original=pc.infer, profile=profile)
    # fast.main retains its validated GPU cache and dispatches child workers here.
    fast.MODULE = MODULE
    original_status = fast.status_lines
    def status_lines(output):
        return [('Resume      bash scripts/run_bsu_allocation_recovery.sh run; verified stages/checkpoints reused'
                 if row.startswith('Resume ') else row) for row in original_status(output)]
    fast.status_lines = status_lines
    fast.main()


if __name__ == '__main__':
    main()
