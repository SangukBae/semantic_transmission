import copy
import itertools
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from semantic_transmission import sparc_selector_budget as v3
from semantic_transmission import sparc_selector_risk as risk
from semantic_transmission import sparc_selector_search_v4 as search
from semantic_transmission import sparc_selector_v3_data as old
from semantic_transmission import sparc_selector_v4_data as data
from semantic_transmission import sparc_selector_train_v4 as training
from semantic_transmission import sparc_selector_train as supervision
from semantic_transmission import sparc_selector_v4 as runner
from semantic_transmission.sparc_selector_data import read, save, sha

GOAL = dict(maximum_psnr_drop_db=.5, maximum_lpips_increase=.01)
SEARCH = dict(beam=4, per_round=2, initial_per_clip=2, measurements_per_clip=6,
              budget_multipliers=[.75,1,1.5,2], risk_ratio=2., risky_max_gap=4)


def observation(indices, rate, psnr=30, lpips=.2):
    return dict(identity=dict(indices=list(indices)), channel=dict(complex_channel_uses=rate),
        quality=dict(mse=10**(-psnr/10), lpips_vgg=lpips, psnr_db=psnr,
                     local_mae95=.1, temporal_mse=.01))


class RiskTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        supervision.STOP=False

    def test_local_context_gradients_and_physical_budget_independence(self):
        torch.manual_seed(2)
        model=risk.RiskSelector(feature_dim=8,width=16,heads=2,local_layers=1,global_layers=1,window=4,dropout=0)
        patches, motion=torch.randn(10,15,8),torch.randn(10,19)
        c,g,s=v3.contexts(10,[[0,4,5,9]]*2,[0,9],[5],24,'cpu')
        out=model(patches,motion,c,g,s,torch.tensor([[0.,0.],[1.,1.]]))
        for key in ('absolute','delta','local_log_mse','violation_logit'):
            torch.testing.assert_close(out[key][0],out[key][1])
        self.assertNotEqual(float(out['score'][0]),float(out['score'][1]))
        self.assertEqual(out['local_log_mse'].shape,(2,10,15))
        sum(v.square().mean() for v in out.values()).backward()
        for part in (model.visual,model.local,model.delta,model.local_error,model.violation):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in part.parameters() if p.grad is not None),0)
        c2,g2,s2=v3.contexts(10,[[0,2,4,5,9]],[0,9],[5],24,'cpu')
        different=model(patches,motion,c2,g2,s2,torch.zeros(1,2))
        self.assertFalse(torch.allclose(out['local_log_mse'][0],different['local_log_mse'][0]))

    def test_signed_deltas_and_pessimistic_calibration(self):
        records=[dict(parent=str(i),actual_rate=2,rate=1,rate_log_sigma=.5,
            actual_delta=[-2,.1],delta=[-1,.02],delta_sigma=[.5,.02],local_log_underprediction=.3) for i in range(10)]
        cal=risk.fit_calibration(records,.1)
        values=dict(rate=torch.ones(1),rate_log_sigma=torch.tensor([.5]),delta=torch.tensor([[-1.,.02]]),
                    delta_sigma=torch.tensor([[.5,.02]]),local_mse=torch.ones(1,2,15))
        corrected=risk.conservative(values,cal)
        self.assertAlmostEqual(float(corrected['rate'][0]),2,places=5)
        self.assertAlmostEqual(float(corrected['psnr'][0]),-2,places=5)
        self.assertAlmostEqual(float(corrected['lpips'][0]),.1,places=5)
        with self.assertRaises(ValueError):risk.fit_calibration(records,.1,split='test')
        with self.assertRaises(ValueError):risk.fit_calibration([records[0]]*10,.1)

    def test_quality_feasible_allocation_matches_bruteforce_nonadditive_costs(self):
        baseline=observation([0,2,4],100)
        clips=[dict(id=str(i),pixels=100,baseline=baseline,
            observations=[baseline,observation([0,4],20,20,.5),observation([0,1,4],70,29.8,.205),
                          observation([0,1,2,4],60,30.1,.19)]) for i in range(2)]
        result=search.allocate_cohort(clips,.21,GOAL)
        choices=[x for x in itertools.product(*(c['observations'] for c in clips))
                 if sum(o['channel']['complex_channel_uses'] for o in x)<=126
                 and all(risk.quality_feasible(o,baseline,GOAL) for o in x)]
        self.assertTrue(result['feasible'])
        self.assertAlmostEqual(result['quality_cost'],min(sum(v3.quality_cost(o,baseline) for o in x)/2 for x in choices))
        impossible=search.allocate_cohort(clips,.1,GOAL)
        self.assertFalse(impossible['feasible']);self.assertEqual(impossible['minimum_observed_safe_uses'],120)
        self.assertEqual(impossible['selected'],[])

    def test_risky_removals_blocked_and_scene_keys_retained(self):
        proxies=np.zeros((20,6,10,3),np.uint8)
        ids=[0,3,6,9,10,13,16,19]
        high=np.ones(20);high[1:9]=10
        candidates=search.risk_proposals(proxies,ids,[0,6,19],[10],high,risk_ratio=2,risky_max_gap=4)
        self.assertTrue(candidates)
        for s in candidates:
            self.assertTrue({0,6,9,10,19}.issubset(s))
            self.assertFalse(any(b-a>4 and high[a+1:b].max(initial=0)>2 for a,b in zip(s,s[1:])))

    def test_feedback_obeys_global_measurement_budget_and_reports_infeasibility(self):
        proxies=np.zeros((20,6,10,3),np.uint8)
        calls=[]
        def measure(ids):
            calls.append(ids)
            return observation(ids,50+len(ids)*10,30 if len(ids)>=5 else 20,.2 if len(ids)>=5 else .5)
        sessions=[search.Session(str(i),proxies,[0,19],[],100,measure,SEARCH,GOAL) for i in range(2)]
        result=search.search_cohort(sessions,.1,SEARCH,GOAL)
        self.assertEqual(len(calls),result['measurement_attempts'])
        self.assertLessEqual(len(calls),result['measurement_cap'])
        self.assertGreater(result['feedback_rounds'],0)
        self.assertFalse(result['cohort']['feasible'])

    def test_spatial_labels_find_corruption_in_correct_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames=np.zeros((2,12,20,3),np.uint8)
            damaged=frames.copy();damaged[1,:4,:4]=255
            path=Path(tmp)/'movie.mkv'
            subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size','20x12',
                '-framerate','24','-i','-','-c:v','ffv1',str(path)],input=damaged.tobytes(),check=True)
            labels=data.spatial_errors(frames,path)
            self.assertEqual(labels.shape,(2,15))
            self.assertGreater(labels[1,0],labels[1,1]+10)
            self.assertTrue(np.all(labels[0] < -10))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        supervision.STOP=False

    def fixture(self,root,device='cpu'):
        rng=np.random.default_rng(5)
        rows=[dict(id=name,parent=name,split=split,dataset='fixture',duration_seconds=1)
              for name,split in [('a','train'),('b','train')]+[(f'c{i}','val') for i in range(11)]+[('test','test')]]
        p=dict(seed=42,signature='fixture',target_cbr=.05,training_budgets=[.03,.05,.08],features={'fixture':True},
            model=dict(feature_dim=8,motion_dim=19,width=16,heads=2,local_layers=1,global_layers=1,window=4,dropout=.1),
            train=dict(device=device,max_epochs=3,patience=3,validation_every=1,states_per_clip=2,accumulate=2,
                       lr=.001,weight_decay=.0001,warmup_epochs=1),quality_goal=GOAL,
            calibration=dict(clips=10,alpha=.1),search=dict(SEARCH,validation_clips=1,test_clips=1),
            loss=dict(absolute=1.,delta=1.,variance=.05,violation=.5,local=.5,ranking=.5,local_tail_fraction=.2))
        p.update(data.split_rows(rows,p))
        for row in rows:
            if row['split']!='train' and row['id'] not in p['calibration_ids']:
                continue  # Neither model-selection nor test labels may be loaded by training.
            folder=root/'clips'/row['id'];folder.mkdir(parents=True)
            proxies=np.zeros((8,6,10,3),np.uint8)
            baseline=observation(v3.seeds(proxies,[0,7],[])[0],150)
            observations=[baseline]+[observation([0,i,7],80+i*10,28+i/2,.22-i*.005) for i in range(1,7)
                                     if [0,i,7]!=baseline['identity']['indices']]
            save(folder/'prepared.json',dict(keys=[0,7],cuts=[],source=dict(video=dict(frames=8,height=6,width=10,fps=24))))
            save(folder/'observations.json',observations);save(folder/'baseline.json',baseline)
            old.atomic_npz(folder/'features.npz',patches=rng.normal(size=(8,15,8)).astype(np.float16),
                motion=rng.normal(size=(8,19)).astype(np.float32),proxies=proxies)
            old.atomic_npz(folder/'risk_labels.npz',local_log_mse=rng.normal(-5,1,(len(observations),8,15)).astype(np.float32))
            names=('prepared.json','features.npz','observations.json','baseline.json','risk_labels.npz')
            save(folder/'risk_inputs.json',dict(schema='sparc_risk_labels_v4',protocol='fixture',
                baseline_identity=baseline['identity'],measurements=len(observations),
                files={name:sha(folder/name) for name in names}))
        return rows,p

    def test_labels_only_verifies_artifacts_and_never_trains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,p=self.fixture(root)
            (root/'.local').mkdir();p.update(source_root=str(root),runtimes={})
            args=SimpleNamespace(root=root,parent=root,source_root=None,config=root/'config.json',
                                 runtimes=None,max_steps=None,labels_only=True)
            with mock.patch.object(runner,'REPO',root),mock.patch.object(runner,'initialize',return_value=(p,rows)), \
                 mock.patch.object(data,'prepare'),mock.patch.object(training,'train') as train, \
                 mock.patch.object(training,'evaluate') as evaluate,mock.patch.object(runner.signal,'signal'), \
                 mock.patch.object(supervision,'runtimes'):
                result=runner.run(args)
                self.assertEqual(result['status'],'LABELS_PREPARED')
                self.assertEqual(result['clips'],12)
                self.assertEqual(result['train_clips'],2)
                self.assertFalse(result['training_started'])
                self.assertTrue(read(root/'run_options.json')['labels_only'])
                self.assertEqual(read(root/'status.json')['done'],12)
                self.assertFalse((root/'RESULT.json').exists())
                train.assert_not_called();evaluate.assert_not_called()
                # Resumed preparation must reject changed labels, even after a previous completion.
                (root/'clips/a/risk_labels.npz').write_bytes(b'corrupted')
                with self.assertRaisesRegex(ValueError,'cache changed'):runner.run(args)
                self.assertEqual(read(root/'status.json')['status'],'FAILED')

    def test_labels_only_mode_survives_resume_and_requires_explicit_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'run_options.json',dict(labels_only=True))
            for flags,expected in [([],True),(['--no-labels-only'],False)]:
                with mock.patch('sys.argv',['v4','launch','--root',str(root),*flags]), \
                     mock.patch.object(runner,'launch',return_value={}) as launch,mock.patch('builtins.print'):
                    runner.main()
                self.assertIs(launch.call_args.args[0].labels_only,expected)

    def test_launcher_forwards_labels_only_to_detached_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args=SimpleNamespace(root=root,parent=root,source_root=None,config=root/'config.json',
                                 runtimes=None,max_steps=None,labels_only=True)
            with mock.patch.object(runner,'initialize'),mock.patch.object(runner.subprocess,'Popen') as popen, \
                 mock.patch.object(runner.distributed,'proc_identity',return_value='test'):
                popen.return_value.pid=123
                self.assertEqual(runner.launch(args)['status'],'LAUNCHED')
                self.assertIn('--labels-only',popen.call_args.args[0])
                self.assertTrue(popen.call_args.kwargs['start_new_session'])

    @staticmethod
    def validate(*args):
        torch.rand(10)
        return dict(selection_score=.5)

    def test_exact_resume_with_calibration_and_new_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b';rows,p=self.fixture(a);self.fixture(b)
            training.train(a,a,rows,p,lambda *a,**k:None,max_steps=5,validate=self.validate)
            training.train(b,b,rows,p,lambda *a,**k:None,max_steps=1,validate=self.validate)
            training.train(b,b,rows,p,lambda *a,**k:None,max_steps=5,validate=self.validate)
            x=torch.load(a/'training/last.pt',weights_only=False);y=torch.load(b/'training/last.pt',weights_only=False)
            self.assertTrue(all(torch.equal(x['model'][k],y['model'][k]) for k in x['model']))
            for key in ('steps','position','epoch','scheduler','best','bad','order','data_hash'):
                self.assertEqual(x[key],y[key],key)
            for key in ('torch_rng','sampler_rng'):self.assertTrue(torch.equal(x[key],y[key]))
            for index,values in x['optimizer']['state'].items():
                for key,value in values.items():torch.testing.assert_close(value,y['optimizer']['state'][index][key],rtol=0,atol=0)
            best=torch.load(a/'training/best.pt',weights_only=False)
            self.assertIn('calibration',best)
            self.assertFalse(set(best['calibration']['parents']) & {r['parent'] for r in rows if r['id'] in p['validation_ids']})

    def test_validation_interruption_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b';rows,p=self.fixture(a);self.fixture(b)
            training.train(a,a,rows,p,lambda *a,**k:None,max_steps=5,validate=self.validate)
            def interrupted(*args):
                torch.rand(10);raise InterruptedError('test interruption')
            with self.assertRaises(InterruptedError):training.train(b,b,rows,p,lambda *a,**k:None,validate=interrupted)
            training.train(b,b,rows,p,lambda *a,**k:None,max_steps=5,validate=self.validate)
            x=torch.load(a/'training/last.pt',weights_only=False);y=torch.load(b/'training/last.pt',weights_only=False)
            self.assertTrue(all(torch.equal(x['model'][k],y['model'][k]) for k in x['model']))
            (a/'clips/a/baseline.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'cache changed'):data.Corpus(a,rows,p)

    def test_parent_leakage_and_fixed_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,p=self.fixture(root)
            rows[-1]['parent']='a'
            with self.assertRaisesRegex(ValueError,'leakage'):data.Corpus(root,rows,p)

    def test_full_evaluation_path_uses_calibration_and_reports_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,p=self.fixture(root);p['target_cbr']=.5;p['supervision_protocol']={}
            checkpoint=root/'candidate.pt'
            model=risk.RiskSelector(**p['model'])
            corpus=data.Corpus(root,rows,p)
            torch.save(dict(architecture=risk.ARCHITECTURE,protocol=p['signature'],model=model.state_dict(),
                normalization=corpus.normalization,calibration=dict(schema='sparc_risk_calibration_v1',
                    parents=[r['parent'] for r in corpus.rows['calibration']],offsets=[0,0,0,0])),checkpoint)
            class Extractor:
                identity={'fixture':True}
                def __init__(self,*_):pass
                def extract(self,frames):
                    return dict(patches=np.ones((8,15,8),np.float16),motion=np.ones((8,19),np.float32),
                                proxies=np.zeros((8,6,10,3),np.uint8))
            class Measurer:
                pixels=480
                def __init__(self,output,source,row,p):
                    folder=output/'clips'/row['id'];folder.mkdir(parents=True)
                    self.array=folder/'source.npy';self.frames=np.zeros((8,6,10,3),np.uint8)
                    np.save(self.array,self.frames)
                    self.ready=dict(keys=[0,7],cuts=[],source=dict(video=dict(frames=8,height=6,width=10,fps=24)))
                def __call__(self,ids):return observation(ids,100,30,.2)
            with mock.patch.object(old,'Extractor',Extractor),mock.patch.object(old,'Measurer',Measurer):
                result=training.evaluate(root,root,rows,p,checkpoint,'test','final',lambda *a,**kw:None)
            self.assertTrue(result['target_met'])
            self.assertTrue(result['summary']['per_video_quality_met'])
            self.assertLessEqual(result['measurement_attempts'],result['measurement_cap'])
            self.assertEqual(result['summary']['selected_videos'],1)
            p['test_ids']=[p['calibration_ids'][0]]
            with self.assertRaisesRegex(ValueError,'leakage'):
                training.evaluate(root,root,rows,p,checkpoint,'test','leaked',lambda *a,**kw:None)


if __name__=='__main__':unittest.main()
