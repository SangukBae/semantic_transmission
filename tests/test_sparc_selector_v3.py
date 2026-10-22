import copy
import itertools
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from semantic_transmission import sparc_selector_budget as b
from semantic_transmission import sparc_selector_train as supervision
from semantic_transmission import sparc_selector_train_v3 as train
from semantic_transmission import sparc_selector_v3_data as data
from semantic_transmission.sparc_selector_data import save, sha


def observation(indices, rate, error):
    return dict(identity=dict(indices=list(indices)), channel=dict(complex_channel_uses=rate),
        quality=dict(mse=error, lpips_vgg=error*2, temporal_mse=error/2,
                     local_mae95=math.sqrt(error), psnr_db=-10*math.log10(error)))


class BudgetTests(unittest.TestCase):
    def test_actual_cost_overrides_optimistic_rank_and_no_fallback(self):
        frames = np.zeros((16, 8, 12, 3), np.uint8)
        measured = []
        def evaluate(ids):
            measured.append(ids)
            return observation(ids, 100, .01)
        decision, _, _ = b.search(frames, [0,15], [], 50, evaluate,
                                  rank=lambda s, _: [1e9]*len(s), rounds=2)
        self.assertFalse(decision['feasible'])
        self.assertIsNone(decision['indices'])
        self.assertTrue(measured)
        self.assertAlmostEqual(b.cbr(observation([0,1], 30, .01), 100), .1)

    def test_joint_moves_preserve_cuts_and_semantic_anchors(self):
        frames = np.zeros((24, 8, 12, 3), np.uint8)
        chosen = list(range(24))
        candidates = b.proposals(frames, chosen, [0,5,23], [12])
        mandatory = {0,5,11,12,23}
        self.assertTrue(any(len(set(chosen)-set(s)) >= 4 for s in candidates))
        self.assertTrue(all(mandatory <= set(s) for s in candidates))

    def test_measured_nonmonotonic_wire_cost_is_respected(self):
        frames = np.zeros((10,8,12,3),np.uint8)
        # Additional anchors can reduce actual cost; never infer cost from cardinality.
        def measure(ids):
            return observation(ids, 120 if len(ids)==2 else 80, .01)
        selected, _, _ = b.search(frames,[0,9],[],100,measure,rounds=1)
        self.assertTrue(selected['feasible'])
        self.assertGreater(len(selected['indices']),2)

    def test_aggregate_budget_matches_brute_force_and_can_fund_hard_video(self):
        rng = np.random.default_rng(23)
        for _ in range(8):
            clips = []
            for i in range(3):
                options = [observation([0,j+1,9], int(rng.integers(10,80)), float(rng.uniform(.01,.1)))
                           for j in range(4)]
                clips.append(dict(id=str(i),pixels=100,baseline=options[0],observations=options))
            result = train.allocate_cohort(clips, .15)
            combinations = [s for s in itertools.product(*(c['observations'] for c in clips))
                            if sum(o['channel']['complex_channel_uses'] for o in s)<=135]
            self.assertEqual(result['feasible'],bool(combinations))
            if combinations:
                optimum = min(sum(b.quality_cost(o,c['baseline']) for o,c in zip(s,clips))/3 for s in combinations)
                self.assertAlmostEqual(result['quality_cost'],optimum)
                self.assertLessEqual(result['channel_uses'],135)


class ModelTests(unittest.TestCase):
    def test_physical_predictions_do_not_change_with_budget_but_policy_does(self):
        torch.set_num_threads(2); torch.manual_seed(11)
        model = b.BudgetSelector(feature_dim=8,width=16,heads=2,local_layers=1,global_layers=1,window=4,dropout=0).eval()
        patches, motion = torch.randn(10,15,8), torch.randn(10,19)
        c,g,s = b.contexts(10,[[0,4,5,9]]*2,[0,9],[5],24,'cpu')
        budget = torch.tensor([[0.,0.],[1.,1.]])
        y,var,score = model(patches,motion,c,g,s,budget)
        torch.testing.assert_close(y[0],y[1],atol=1e-6,rtol=1e-5)
        self.assertNotEqual(float(score[0]),float(score[1]))
        (y.square().sum()+score.square().sum()+var.square().sum()).backward()
        for part in (model.visual,model.local,model.global_time,model.policy,model.distortion_rate):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in part.parameters() if p.grad is not None),0)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        supervision.STOP=False

    def fixture(self,root,device='cpu'):
        rng=np.random.default_rng(7); rows=[]
        for name,split in [('a','train'),('b','train'),('c','val'),('d','test')]:
            row=dict(id=name,parent=name,dataset='fixture',split=split,duration_seconds=1)
            rows.append(row)
            if split=='test':
                continue  # No test feature/target file can be read during training.
            folder=root/'clips'/name;folder.mkdir(parents=True)
            observations=[observation([0,i,7],100+i*20,.1/(i+1)) for i in range(1,7)]
            save(folder/'prepared.json',dict(keys=[0,7],cuts=[],source=dict(video=dict(frames=8,height=8,width=12,fps=24.))))
            save(folder/'base_observations.json',observations)
            data.atomic_npz(folder/'features.npz',patches=rng.normal(size=(8,15,8)).astype(np.float16),
                            motion=rng.normal(size=(8,19)).astype(np.float32))
            save(folder/'inputs.json',dict(features_sha256=sha(folder/'features.npz'),
                observations_sha256=sha(folder/'base_observations.json')))
        p=dict(seed=42,signature='fixture',target_cbr=.05,training_budgets=[.03,.05,.08],features={'fixture':True},
            model=dict(feature_dim=8,width=16,heads=2,local_layers=1,global_layers=1,window=4,dropout=.1),
            train=dict(device=device,max_epochs=4,patience=3,validation_every=1,states_per_clip=2,
                       accumulate=2,lr=.001,weight_decay=.0001,warmup_epochs=1))
        return rows,p

    @staticmethod
    def validate(*args):
        # Simulate evaluators that consume RNG, proving they cannot perturb training resume.
        torch.rand(20)
        return dict(selection_score=.5)

    def test_exact_resume_and_sealed_test_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,bb=Path(tmp)/'a',Path(tmp)/'b'
            rows,p=self.fixture(a);self.fixture(bb)
            train.train(a,a,rows,p,lambda *a,**k:None,max_steps=4,validate=self.validate)
            train.train(bb,bb,rows,p,lambda *a,**k:None,max_steps=1,validate=self.validate)
            train.train(bb,bb,rows,p,lambda *a,**k:None,max_steps=4,validate=self.validate)
            x=torch.load(a/'training/last.pt',weights_only=False)
            y=torch.load(bb/'training/last.pt',weights_only=False)
            for key in x['model']:
                self.assertTrue(torch.equal(x['model'][key],y['model'][key]),key)
            for key in ('steps','position','epoch','scheduler','best','bad','data_hash','order'):
                self.assertEqual(x[key],y[key],key)
            for key in ('torch_rng','sampler_rng'):
                self.assertTrue(torch.equal(x[key],y[key]))
            for idx,values in x['optimizer']['state'].items():
                for key,value in values.items():
                    torch.testing.assert_close(value,y['optimizer']['state'][idx][key],rtol=0,atol=0)
            self.assertTrue((a/'training/best.pt').exists())

    def test_source_parent_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,p=self.fixture(root)
            rows[-1]['parent']=rows[0]['parent']
            with self.assertRaisesRegex(ValueError,'leakage'):
                data.Corpus(root,rows,p)

    def test_interruption_during_validation_preserves_next_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,bb=Path(tmp)/'a',Path(tmp)/'b'
            rows,p=self.fixture(a);self.fixture(bb)
            train.train(a,a,rows,p,lambda *a,**k:None,max_steps=4,validate=self.validate)
            def interrupted(*args):
                torch.rand(20)
                raise InterruptedError('validation interrupted')
            with self.assertRaises(InterruptedError):
                train.train(bb,bb,rows,p,lambda *a,**k:None,validate=interrupted)
            train.train(bb,bb,rows,p,lambda *a,**k:None,max_steps=4,validate=self.validate)
            x=torch.load(a/'training/last.pt',weights_only=False)
            y=torch.load(bb/'training/last.pt',weights_only=False)
            self.assertTrue(all(torch.equal(x['model'][k],y['model'][k]) for k in x['model']))
            self.assertTrue(torch.equal(x['torch_rng'],y['torch_rng']))


class DispatchTests(unittest.TestCase):
    def test_old_optimizer_is_replaced_and_callbacks_are_restored(self):
        from semantic_transmission import sparc_selector_dispatch as dispatcher
        from semantic_transmission import sparc_selector_train_v2 as old_v2
        original_train, original_build = old_v2.train, old_v2.build_inputs
        def worker(root, plan, role, runtimes):
            old_v2.build_inputs(root, [])
            return old_v2.train(root, {}, [])
        with mock.patch.object(dispatcher.d,'worker',side_effect=worker), mock.patch.object(
                dispatcher,'dispatch',return_value={'v3':True}) as callback:
            self.assertEqual(dispatcher.run(Path('/unused'),Path('/unused')) ,{'v3':True})
            callback.assert_called_once()
        self.assertIs(old_v2.train,original_train)
        self.assertIs(old_v2.build_inputs,original_build)

    def test_changed_release_config_cannot_launch_training(self):
        from semantic_transmission import sparc_selector_dispatch as dispatcher
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'config.json';save(config,{'a':1})
            save(root/'.distributed/training_route.json',dict(enabled=True,schema='sparc_training_route_v3',
                code={},config=str(config),config_sha256='invalid'))
            with self.assertRaisesRegex(ValueError,'settings changed'):
                dispatcher.dispatch(root,{},[])


if __name__=='__main__':
    unittest.main()
