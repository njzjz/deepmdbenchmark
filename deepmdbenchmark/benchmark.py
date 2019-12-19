import json
import os

import numpy as np
from deepmd.Trainer import NNPTrainer
from deepmd.RunOptions import RunOptions
from deepmd.common import data_requirement
from deepmd.DataSystem import DeepmdDataSystem



class Benchmark:
    def __init__(self):
        # setup
        with open(os.path.join(os.path.dirname(__file__), 'water_se_a.json')) as fp:
            jdata = json.load(fp)
        run_opt = RunOptions(None)
        run_opt.verbose = False
        self.model = NNPTrainer(jdata, run_opt=run_opt)
        rcut = self.model.model.get_rcut()
        type_map = self.model.model.get_type_map()
        systems = os.path.join(os.path.dirname(__file__), 'data')
        set_pfx = jdata['training']['set_prefix']
        seed = jdata['training']['seed']

        np.random.seed(seed)
        batch_size = jdata['training']['batch_size']
        test_size = jdata['training']['numb_test']
        self.data = DeepmdDataSystem(systems, batch_size, test_size, rcut,
                                set_prefix=set_pfx, run_opt=run_opt, type_map=type_map)
        self.data.add_dict(data_requirement)
        self.model.build(self.data)
        self.model._init_sess_serial()

        cur_batch = self.model.sess.run(self.model.global_step)
        self.cur_batch = cur_batch


    def train_step(self):
        batch_data = self.data.get_batch (sys_weights = self.model.sys_weights)
        feed_dict_batch = {}
        for kk in batch_data.keys():
            if kk == 'find_type' or kk == 'type' :
                continue
            if 'find_' in kk :
                feed_dict_batch[self.model.place_holders[kk]] = batch_data[kk]
            else:
                feed_dict_batch[self.model.place_holders[kk]] = np.reshape(batch_data[kk], [-1])
        for ii in ['type'] :
            feed_dict_batch[self.model.place_holders[ii]] = np.reshape(batch_data[ii], [-1])
        for ii in ['natoms_vec', 'default_mesh'] :
            feed_dict_batch[self.model.place_holders[ii]] = batch_data[ii]
        feed_dict_batch[self.model.place_holders['is_training']] = True

        self.model.sess.run([self.model.train_op], feed_dict = feed_dict_batch, options=None, run_metadata=None)
        self.model.sess.run(self.model.global_step)

def test_performance(benchmark):
    b=Benchmark()
    benchmark(b.train_step)
