import copy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from semantic_transmission import sparc_selector_webvid_search as w
from semantic_transmission.sparc_selector_data import read,save


def obs(indices, rate, mse=.01, lpips=.1):
    return dict(identity=dict(indices=list(indices)),channel=dict(complex_channel_uses=rate),
        quality=dict(mse=mse,psnr_db=-10*math.log10(mse),lpips_vgg=lpips,
                     temporal_mse=mse/2,local_mae95=math.sqrt(mse)))


class SourceTests(unittest.TestCase):
    def test_train_only_longest_first_and_duplicate_windows_removed(self):
        def row(id,parent,split,length,start=0):
            return dict(id=id,parent=parent,split=split,dataset='WebVid-existing',duration_seconds=length,start_seconds=start)
        rows=[row('a8','a','train',8),row('a12','a','train',12),row('b8','b','train',8),
              row('b8duplicate','b','train',8),row('short','c','train',4),row('val','d','val',12),row('test','e','test',12)]
        selected=w.select_rows(rows,dict(minimum_seconds=8,windows_per_parent=2))
        self.assertEqual([r['id'] for r in selected],['a12','b8','a8'])

    def test_parent_requires_successful_completion_and_every_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=dict(signature='p',assignments=dict(local=['a','b']))
            path=root/'.distributed/worker_local.json'
            save(path,dict(status='LABELS_COMPLETE',plan='p',exit_code=0))
            save(root/'clips/a/labels.json',{})
            self.assertFalse(w.parent_completion(root,plan)[0])
            save(root/'clips/b/label_failure.json',{})
            self.assertTrue(w.parent_completion(root,plan)[0])
            save(path,dict(status='INTERRUPTED',plan='p',exit_code=130))
            self.assertFalse(w.parent_completion(root,plan)[0])
            save(path,dict(status='LABELS_COMPLETE',plan='different',exit_code=0))
            self.assertFalse(w.parent_completion(root,plan)[0])


class SearchTests(unittest.TestCase):
    def config(self):
        return dict(budgets=[.0005,.00065,.0008,.001],target_cbr=.00065,rounds=3,per_round=4,
                    beam_per_budget=2,maximum_new_candidates_per_clip=10,
                    maximum_psnr_drop_db=.5,maximum_lpips_increase=.01)

    def test_quality_preserving_candidate_wins_and_rate_is_measured(self):
        cfg=self.config();cfg['budgets']=[1.]
        base=obs([0,3,7],400)
        bad=obs([0,7],200,.02,.05) # cheaper but PSNR fails
        good=obs([0,4,7],290,.0101,.105)
        extra=obs([0,2,5,7],301,.005,.05) # quality good but over budget
        result=w.measured_summary([base,bad,good,extra],base,100,cfg)['decisions']['1.0']
        self.assertTrue(result['quality_goal_met'])
        self.assertEqual(result['indices'],good['identity']['indices'])
        self.assertLessEqual(result['channel_uses'],300)

    def test_no_overbudget_fallback_or_false_quality_success(self):
        cfg=self.config();base=obs([0,4,7],400)
        result=w.measured_summary([base],base,100,cfg)
        for decision in result['decisions'].values():
            self.assertFalse(decision['feasible']);self.assertFalse(decision['quality_goal_met'])
            self.assertIsNone(decision['indices'])

    def test_interrupted_candidate_journal_resumes_same_search_with_hard_cap(self):
        rng=np.random.default_rng(23);frames=rng.integers(0,256,(24,8,12,3),dtype=np.uint8)
        cfg=self.config();calls=[]
        def measure(indices):
            calls.append(list(indices))
            # Deliberately non-monotonic bit cost; no linear frame-cost assumption.
            return obs(indices,100+17*(sum(indices)%7),.01+.001*(24-len(indices)))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=w.search(root/'full.json',dict(frozen='a'),frames,[0,23],[12],100000,measure,[],cfg)
            complete_calls=copy.deepcopy(calls);calls.clear()
            count=[0]
            def interrupted(indices):
                if count[0]==3:raise InterruptedError('simulated shutdown before committing candidate')
                count[0]+=1;return measure(indices)
            with self.assertRaises(InterruptedError):
                w.search(root/'resume.json',dict(frozen='a'),frames,[0,23],[12],100000,interrupted,[],cfg)
            b=w.search(root/'resume.json',dict(frozen='a'),frames,[0,23],[12],100000,measure,[],cfg)
            self.assertEqual(a,b);self.assertEqual(calls,complete_calls)
            self.assertEqual(read(root/'full.json'),read(root/'resume.json'))
            self.assertLessEqual(len(calls),cfg['maximum_new_candidates_per_clip'])
            self.assertEqual(len(calls),len({tuple(s) for s in calls}))
            self.assertTrue(all({0,11,12,23}<=set(s) for s in calls))
            with self.assertRaisesRegex(ValueError,'identity'):
                w.search(root/'resume.json',dict(frozen='b'),frames,[0,23],[12],100000,measure,[],cfg)

    def test_failed_baseline_is_retained_and_excluded(self):
        frames=np.zeros((16,8,12,3),dtype=np.uint8);cfg=self.config()
        baseline=w.budget.seeds(frames,[0,15],[])[0]
        def measure(indices):return None if tuple(indices)==baseline else obs(indices,10)
        with tempfile.TemporaryDirectory() as tmp:
            result=w.search(Path(tmp)/'journal.json',dict(a=1),frames,[0,15],[],100000,measure,[],cfg)
            self.assertEqual(result['status'],'BASELINE_CHANNEL_REJECTED')
            self.assertFalse(result['use_for_training'])

    def test_new_candidate_cap_excludes_reused_observations(self):
        frames=np.zeros((16,8,12,3),dtype=np.uint8);cfg=self.config()
        initial=w.budget.seeds(frames,[0,15],[])
        known=[obs(s,100) for s in initial];calls=[]
        def measure(indices):calls.append(indices);return obs(indices,90)
        with tempfile.TemporaryDirectory() as tmp:
            w.search(Path(tmp)/'journal.json',dict(a=1),frames,[0,15],[],100000,measure,known,cfg)
        self.assertTrue(all(tuple(s) not in initial for s in calls))
        self.assertLessEqual(len(calls),cfg['maximum_new_candidates_per_clip'])


if __name__=='__main__':unittest.main()
