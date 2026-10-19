"""The BSU-Gen Qwen selector, with source-bound replay and preserved call caches."""
from pathlib import Path
import shutil

from . import auto_extraction as ae
from . import qwen35_semantic_selection as semantic
from .qwen35_backend import MODEL_ID, REVISION, SNAPSHOT
from .qwen35_state_delta import Session
from .qwen35_feedback import Session as BaseSession
from .artifacts import sha256, write_json
from .webvid5 import fingerprint

LENGTH_ERROR = 'provide two short visible state descriptions, at most 45 words total'


def concise_prompt(prompt):
    return prompt + ('\nFORMAT REPAIR: Validation error: ' + LENGTH_ERROR +
        '. Write reference_state and candidate_state with at most 16 words EACH '
        '(at most 32 words combined). Keep all required fields and actual frame IDs. '
        'Reinspect the images; do not change a semantic decision just to satisfy formatting.')


def identity(source, config):
    return dict(policy=semantic.POLICY, model_id=MODEL_ID, revision=REVISION,
        quantization='nf4', candidate_stride=config['candidate_stride'], max_pixels=262144,
        tokens=256, prompt=semantic.PROMPT, video=source['video'], frame_hashes=source['hashes'])


def validate_manifest(path, source, config):
    import json
    manifest = json.loads(path.read_text())
    if (manifest.get('identity') != identity(source, config) or
            manifest.get('checksum') != fingerprint({k:v for k,v in manifest.items() if k != 'checksum'})):
        raise ValueError('semantic replay source/policy/checksum differs')
    semantic.validate_keys(manifest['keys'], source['video']['frames'])
    if not manifest['cache_files'] or not manifest['calls']:
        raise ValueError('semantic replay lacks actual Qwen call evidence')
    for name, digest in manifest['cache_files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('invalid semantic cache path')
        if sha256(path.parent / 'semantic_cache' / relative) != digest:
            raise ValueError('semantic replay call artifact changed')
    return manifest


def select(source, config, output, replay=None, progress=None):
    output = Path(output); cache = output / 'semantic_cache'
    for name, digest in source['hashes'].items():
        if sha256(Path(source['directory']) / name) != digest:
            raise ValueError('semantic source frame changed')
    expected = validate_manifest(Path(replay), source, config) if replay else None
    if expected:
        cache.mkdir(exist_ok=True)
        for name, digest in expected['cache_files'].items():
            dest = cache / name; dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if sha256(dest) != digest: raise ValueError('existing replay cache differs')
            else:
                shutil.copy2(Path(replay).parent / 'semantic_cache' / name, dest)
    else:
        cache.mkdir(exist_ok=True)
    ident = identity(source, config)
    plan = dict(source_frames=source['directory'], frames=source['video']['frames'],
        fps=source['video']['fps'], source_frame_hashes={k.removesuffix('.png'):v for k,v in source['hashes'].items()},
        signature=fingerprint(ident), model_id=MODEL_ID, model_revision=REVISION,
        model_path=str(SNAPSHOT), quantization='nf4', max_pixels=262144, max_images=16)
    class VerifiedSession(Session):
        def call(self, role, ids, prompt, validator, tokens=256):
            for index in ids: ae.check_frame(self.plan, index)
            repaired = concise_prompt(prompt)
            for path in (self.root / 'calls' / (role + '_concise')).glob('*.json'):
                record = ae.unseal(path)
                if (record['identity']['ids'] == ids and
                        record['identity']['prompt'].startswith(repaired + '\nVALID SOURCE')):
                    return BaseSession.call(self, role + '_concise', ids, repaired, validator, tokens)
            try:
                return super().call(role, ids, prompt, validator, tokens)
            except RuntimeError as exc:
                if not str(exc).startswith('three invalid responses:') or LENGTH_ERROR not in str(exc):
                    raise
                return BaseSession.call(self, role + '_concise', ids, repaired, validator, tokens)
    def no_new_calls(_):
        raise ValueError('semantic replay is incomplete; new Qwen inference is forbidden in replay mode')
    session = VerifiedSession(cache, plan, factory=no_new_calls) if expected else VerifiedSession(cache, plan)
    keys, decisions = semantic.select(source['video']['frames'], session,
        stride=config['candidate_stride'], tokens=256,
        progress=(lambda n, total, _: progress(n, total)) if progress else None)
    if expected and (keys != expected['keys'] or decisions != expected['decisions']):
        raise ValueError('replayed semantic decisions differ')
    calls = [dict(role=c['role'], path=str(Path(c['path']).relative_to(cache)), sha256=c['sha256'])
             for c in session.calls]
    manifest = dict(identity=ident, keys=keys, decisions=decisions, calls=calls,
        cache_files={str(p.relative_to(cache)):sha256(p) for p in sorted(cache.rglob('*')) if p.is_file()},
        new_model_calls=sum(not c['reused'] for c in session.calls),
        replayed_model_calls=sum(c['reused'] for c in session.calls),
        independent_semantic_review='NOT_PERFORMED')
    manifest['checksum'] = fingerprint(manifest)
    write_json(output / 'semantic.json', manifest)
    return manifest
