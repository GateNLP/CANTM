import sys
from GateMIcateLib import ModelUltiUpdateCAtopic as ModelUlti
from GateMIcateLib import BatchIterBert, DictionaryProcess
from GateMIcateLib.batchPostProcessors import bowBertBatchProcessor as batchPostProcessor
from GateMIcateLib import ScholarPostProcessor as ReaderPostProcessor
from GateMIcateLib.readers import WVmisInfoDataIter as dataIter
from configobj import ConfigObj
from torch.nn import init
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("testReaderargs", help="args for test reader")
    parser.add_argument("--configFile", help="config files if needed")
    parser.add_argument("--cachePath", help="save models")
    parser.add_argument("--randomSeed", type=int, help="randomSeed for reproduction")
    parser.add_argument("--num_epoches", type=int, default=5, help="num epoches")
    parser.add_argument("--patient", type=int, default=40, help="early_stop_patient")
    parser.add_argument("--earlyStopping", default='cls_loss', help="early stopping")
    parser.add_argument("--corpusType", default='wvmisinfo', help="corpus type, for select reader")
    parser.add_argument("--x_fields", help="x fileds", default='Claim,Explaination')
    parser.add_argument("--y_field", help="y filed", default='Label')
    args = parser.parse_args()

    testReaderargs = args.testReaderargs.split(',')
    x_fields = args.x_fields.split(',')

    config = ConfigObj(args.configFile)
    mUlti = ModelUlti(load_path=args.cachePath, gpu=True)
    trainable_weights = ['xy_topics.topic.weight', 
            'z_y_hidden.hidden1.weight',
            'z2y_classifier.layer_output.weight',
            ]

    trainable_bias = [
            'xy_topics.topic.bias',
            'z_y_hidden.hidden1.bias',
            'z2y_classifier.layer_output.bias'
            ]
    trainable_no_init = [
            'mu_z2.weight',
            'mu_z2.bias',
            'log_sigma_z2.weight',
            'log_sigma_z2.bias',
            'x_y_hidden.hidden1.weight',
            'x_y_hidden.hidden1.bias'
            ]


    for name, param in mUlti.net.named_parameters():
        print(name)
        if name in trainable_weights:
            param.requires_grad = True
            param.data.uniform_(-1.0, 1.0)
        elif name in trainable_bias:
            param.requires_grad = True
            param.data.fill_(0)
        elif name in trainable_no_init:
            param.requires_grad = True
        else:
            param.requires_grad = False




    postProcessor = ReaderPostProcessor(config=config, word2id=True, remove_single_list=False, add_spec_tokens=True, x_fields=x_fields, y_field=args.y_field, max_sent_len=300)
    postProcessor.dictProcess = mUlti.bowdict

    testDataIter = dataIter(*testReaderargs, label_field=args.y_field, postProcessor=postProcessor, config=config, shuffle=True)

    testBatchIter = BatchIterBert(testDataIter, filling_last_batch=True, postProcessor=batchPostProcessor, batch_size=32)

    mUlti.train(testBatchIter, num_epohs=args.num_epoches, cache_path=args.cachePath)




