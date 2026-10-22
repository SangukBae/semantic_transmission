from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

from PIL import Image
from semantic_transmission import sparc_webvid_recovery as recovery
from semantic_transmission import auto_extraction as ae
from semantic_transmission.sparc_selector_data import save,sha
from semantic_transmission.qwen35_backend import MODEL_ID,REVISION


class RepairTests(unittest.TestCase):
    def fixture(self,root):
        folder=root/'clips/webvid_006_96620087';frames=folder/'frames';frames.mkdir(parents=True)
        for i in (308,324):Image.new('RGB',(16,16),(i%256,30,40)).save(frames/f'{i}.png')
        identity=dict(plan_signature='fixture',role='semantic_select_schema',ids=[308,324],
            frame_hashes=[sha(frames/f'{i}.png') for i in (308,324)],prompt='Inspect these source images.',
            labels=['SOURCE frame 308','SOURCE frame 324'],tokens=256)
        key=ae.digest(identity)
        invalid=dict(decision='changed',kind='pose',reference_state='person facing away',
            candidate_state='person facing away',evidence_frames=[308,324])
        base=folder/'semantic_cache/failed_attempts/semantic_select_schema'
        for i in range(3):save(base/f'{key}_{i}.json',ae.seal(dict(identity=identity,
            attempt=i,raw_text=json.dumps(invalid),error=recovery.ERROR,result=invalid)))
        save(root/'status.json',dict(status='FAILED',stage='semantic_keys',clip=folder.name,
            error=f'three invalid responses: semantic_select_schema/{key}; last error: {recovery.ERROR}'))
        return folder,key,invalid

    def response(self,value):
        return dict(raw_text=json.dumps(value),truncated=False,seconds=1.,
                    model_runtime=dict(model_id=MODEL_ID,revision=REVISION,quantization='nf4'))

    def test_model_repairs_are_validated_and_original_evidence_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,key,invalid=self.fixture(root)
            before={p:sha(p) for p in folder.glob('semantic_cache/failed_attempts/*/*.json')}
            valid=dict(invalid,decision='same',kind='none')
            model=mock.Mock();model.generate.side_effect=[self.response(invalid),self.response(valid)]
            result=recovery.recover(root,factory=lambda _:model)
            self.assertEqual(result['status'],'REPAIRED');self.assertEqual(result['result'],valid)
            self.assertEqual(model.generate.call_count,2)
            for p,h in before.items():self.assertEqual(sha(p),h)
            record=ae.unseal(Path(result['accepted_response']))
            self.assertEqual(record['result'],ae.parse_json(record['raw_text']))
            self.assertEqual(record['identity']['ids'],[308,324])
            self.assertIn(recovery.ERROR,record['actual_prompt'])
            self.assertTrue(record['recovery']['original_validator_preserved'])
            factory=mock.Mock(side_effect=AssertionError('cache must be reused'))
            self.assertEqual(recovery.recover(root,factory)['status'],'REUSED_REPAIRED_RESPONSE')
            factory.assert_not_called()

    def test_invalid_repeated_outputs_never_become_an_accepted_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,key,invalid=self.fixture(root)
            model=mock.Mock();model.generate.return_value=self.response(invalid)
            with self.assertRaisesRegex(RuntimeError,'exhausted'):
                recovery.recover(root,factory=lambda _:model)
            self.assertEqual(model.generate.call_count,3)
            self.assertFalse(list(folder.glob('semantic_cache/calls/*/*.json')))
            with self.assertRaisesRegex(RuntimeError,'exhausted'):
                recovery.recover(root,factory=lambda _:self.fail('must not retry indefinitely'))

    def test_changed_source_frames_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,_,_=self.fixture(root)
            Image.new('RGB',(16,16),'red').save(folder/'frames/308.png')
            with self.assertRaisesRegex(ValueError,'source frame changed'):
                recovery.recover(root,factory=lambda _:self.fail('must reject before inference'))

    def test_unrelated_failures_are_not_automatically_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            state=json.loads((root/'status.json').read_text());state['error']='CUDA out of memory';save(root/'status.json',state)
            with self.assertRaisesRegex(ValueError,'outside'):
                recovery.failure(root)

    def test_recovery_code_changes_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recovery.policy(root)
            with mock.patch.object(recovery,'sha',return_value='changed'):
                with self.assertRaisesRegex(ValueError,'policy changed'):recovery.policy(root)


if __name__=='__main__':unittest.main()
