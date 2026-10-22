import copy
from pathlib import Path
import tempfile
import unittest

import torch

from semantic_transmission import sparc_selector_budget as budget
from semantic_transmission import sparc_selector_finetune as continuation
from semantic_transmission import sparc_selector_train_v3 as trainer
from semantic_transmission import sparc_selector_train as supervision
from semantic_transmission.sparc_selector_data import sha, read
from test_sparc_selector_v3 import TrainingTests


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        supervision.STOP=False

    def initial(self, root, p):
        torch.manual_seed(900)
        model=budget.BudgetSelector(**p['model'])
        record=dict(architecture=budget.ARCHITECTURE,model_config=p['model'],features=p['features'],
                    protocol='parent',model=model.state_dict(),
                    normalization=dict(mean=torch.arange(5).float(),std=torch.ones(5)*2))
        path=root/'parent.pt';torch.save(record,path)
        p['initialization']=dict(sha256=sha(path),protocol='parent')
        return path,record

    def test_warm_start_preserves_weights_normalization_and_parent_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,p=TrainingTests().fixture(root)
            initial,record=self.initial(root,p)
            supervision.STOP=True
            result=trainer.train(root,root,rows,p,lambda *a,**k:None,
                initial_checkpoint=initial,initial_validation=dict(selection_score=.1))
            saved=torch.load(root/'training/last.pt',weights_only=False)
            self.assertEqual(result['steps'],0)
            self.assertEqual(saved['optimizer']['state'],{})
            self.assertTrue(all(torch.equal(v,saved['model'][k]) for k,v in record['model'].items()))
            self.assertTrue(all(torch.equal(v,saved['normalization'][k]) for k,v in record['normalization'].items()))
            supervision.STOP=False
            trainer.train(root,root,rows,p,lambda *a,**k:None,max_steps=4,
                validate=lambda *a,**k:dict(selection_score=.2),initial_checkpoint=initial,
                initial_validation=dict(selection_score=.1))
            best=torch.load(root/'training/best.pt',weights_only=False)
            self.assertTrue(best['retained_parent'])
            self.assertTrue(all(torch.equal(v,best['model'][k]) for k,v in record['model'].items()))

    def test_warm_start_exact_resume_optimizer_rng_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            rows,p=TrainingTests().fixture(a);TrainingTests().fixture(b)
            initial,_=self.initial(a,p)
            kwargs=dict(initial_checkpoint=initial,initial_validation=dict(selection_score=.1),validate=TrainingTests.validate)
            trainer.train(a,a,rows,p,lambda *a,**k:None,max_steps=4,**kwargs)
            trainer.train(b,b,rows,p,lambda *a,**k:None,max_steps=1,**kwargs)
            trainer.train(b,b,rows,p,lambda *a,**k:None,max_steps=4,**kwargs)
            x=torch.load(a/'training/last.pt',weights_only=False);y=torch.load(b/'training/last.pt',weights_only=False)
            self.assertTrue(all(torch.equal(v,y['model'][k]) for k,v in x['model'].items()))
            for key in ('steps','epoch','position','scheduler','order','best','bad'):
                self.assertEqual(x[key],y[key],key)
            for key in ('torch_rng','sampler_rng'):
                self.assertTrue(torch.equal(x[key],y[key]))
            for index,values in x['optimizer']['state'].items():
                for key,value in values.items():
                    torch.testing.assert_close(value,y['optimizer']['state'][index][key],rtol=0,atol=0)

    def test_changed_initial_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,p=TrainingTests().fixture(root);initial,_=self.initial(root,p)
            initial.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):
                trainer.train(root,root,rows,p,lambda *a,**k:None,initial_checkpoint=initial,
                              initial_validation=dict(selection_score=.1))

    def test_holdout_parent_cannot_enter_long_training_corpus(self):
        a=dict(id='a',parent='shared',split='train');b=dict(id='b',parent='shared',split='test')
        selection=dict(long=[a],replay=[],val=[],test=[b]);dataset=dict(split='train',clips=[dict(input=a)])
        with self.assertRaisesRegex(ValueError,'leakage'):
            continuation.validate_selection(selection,[a,b],dataset)

    def test_real_training_progress_accepts_resume_message_and_epoch_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,p=TrainingTests().fixture(root);initial,_=self.initial(root,p)
            trainer.train(root,root,rows,p,lambda *a,**k:continuation.write_progress(root,*a,**k),
                max_steps=1,initial_checkpoint=initial,initial_validation=dict(selection_score=.1))
            value=read(root/'status.json')
            self.assertEqual(value['step'],1)
            self.assertIn('restored together',value['resume'])
            continuation.write_progress(root,'train_v3',1,4,epoch=2,batch_done=2,batch_total=20)
            self.assertEqual(read(root/'status.json')['epoch_start_batch'],2)


if __name__=='__main__':unittest.main()
