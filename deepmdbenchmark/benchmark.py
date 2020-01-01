import json
import os
import timeit
import logging
import nvgpu

import numpy as np
from deepmd.Trainer import NNPTrainer
from deepmd.RunOptions import RunOptions
import deepmd.cluster.Local as Local
from deepmd.common import data_requirement
from deepmd.DataSystem import DeepmdDataSystem
from cpuinfo import get_cpu_info
import leancloud

import deepmd
import tensorflow as tf


class Benchmark:
    def __init__(self):
        # setup
        with open(os.path.join(os.path.dirname(__file__), 'water_se_a.json')) as fp:
            jdata = json.load(fp)
        self.run_opt = RunOptions(None)
        self.run_opt.verbose = False
        self.model = NNPTrainer(jdata, run_opt=self.run_opt)
        rcut = self.model.model.get_rcut()
        type_map = self.model.model.get_type_map()
        systems = [os.path.join(os.path.dirname(__file__), 'data')]
        set_pfx = jdata['training']['set_prefix']
        seed = jdata['training']['seed']

        np.random.seed(seed)
        batch_size = jdata['training']['batch_size']
        test_size = jdata['training']['numb_test']
        self.data = DeepmdDataSystem(systems, batch_size, test_size, rcut,
                                     set_prefix=set_pfx, run_opt=self.run_opt, type_map=type_map)
        self.data.add_dict(data_requirement)
        self.model.build(self.data)
        self.model._init_sess_serial()

        cur_batch = self.model.sess.run(self.model.global_step)
        self.cur_batch = cur_batch

    def train_step(self):
        batch_data = self.data.get_batch(sys_weights=self.model.sys_weights)
        feed_dict_batch = {}
        for kk in batch_data.keys():
            if kk == 'find_type' or kk == 'type':
                continue
            if 'find_' in kk:
                feed_dict_batch[self.model.place_holders[kk]] = batch_data[kk]
            else:
                feed_dict_batch[self.model.place_holders[kk]
                                ] = np.reshape(batch_data[kk], [-1])
        for ii in ['type']:
            feed_dict_batch[self.model.place_holders[ii]
                            ] = np.reshape(batch_data[ii], [-1])
        for ii in ['natoms_vec', 'default_mesh']:
            feed_dict_batch[self.model.place_holders[ii]] = batch_data[ii]
        feed_dict_batch[self.model.place_holders['is_training']] = True

        self.model.sess.run(
            [self.model.train_op], feed_dict=feed_dict_batch, options=None, run_metadata=None)
        self.model.sess.run(self.model.global_step)


def get_performance(benchmark):
    logging.warning("start test")
    # the first step is a bit slow
    benchmark.train_step()
    timer = timeit.Timer(benchmark.train_step)
    inital_time = timer.timeit(number=10)
    test_times = 600 // inital_time
    logging.warning("test %d times"%(test_times))
    final_times = timer.repeat(repeat=int(test_times), number=1)
    logging.warning("min: %f s max: %f s mean: %f s"%(np.min(final_times), np.max(final_times), np.mean(final_times)))
    return np.min(final_times)

def get_env():
    # deepmd version
    deepmd_version = deepmd.__version__

    # tf version
    try:
        tf_version = tf.version.VERSION 
    except AttributeError:
        tf_version = tf.VERSION
    
    if tf.test.is_gpu_available():
        hardware_type = "gpu"
        _, _, gpus = Local.get_resource()
        hardware_name = nvgpu.gpu_info()[gpus[0]]['type']
    else:
        hardware_type = "cpu"
        cpu_info = get_cpu_info()
        hardware_name = "%s * %s" % (cpu_info['brand'], cpu_info['count'])
    logging.warning("deepmd version:%s tensorflow version:%s hardware:%s"%(deepmd_version, tf_version, hardware_name))
    return deepmd_version, tf_version, hardware_name, hardware_type

def upload(upload_dict):
    logging.warning("uploading...")
    # init leancloud
    leancloud.init("GNtNHs8tmJWnwDHlHnwqMzbm-MdYXbMMI", "XEHFtBRddK1lx5eLXDSsKWfL")
    Todo = leancloud.Object.extend('deepmdbenchmark')
    query = Todo.query
    for key, value in upload_dict.items():
        if key != 'time':
            query.equal_to(key, value)
    try:
        todo = query.first()
        if todo.get('time') <= upload_dict['time']:
            logging.warning("Not break the record")
            return
    except leancloud.errors.LeanCloudError:
        todo = Todo()
    for key, value in upload_dict.items():
        todo.set(key, value)
    todo.save()
    logging.warning("Uploaded")


def run():
    b = Benchmark()
    time = get_performance(b)
    #time = 1
    deepmd_version, tf_version, hardware_name, hardware_type = get_env()
    upload_dict = {
        "time": time,
        "deepmd_version": deepmd_version,
        "tf_version": tf_version,
        "hardware_name": hardware_name,
        "hardware_type": hardware_type
    }
    upload(upload_dict)

if __name__ == '__main__':
    run()
