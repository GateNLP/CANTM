import argparse
import json
import os
from GateMIcateLib import EvaluationManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trainReaderargs", help="args for train reader")
    parser.add_argument("--testReaderargs", help="args for test reader")
    parser.add_argument("--valReaderargs", help="args for val reader")
    parser.add_argument("--configFile", help="config files if needed")
    parser.add_argument("--cachePath", help="save models")
    parser.add_argument("--nFold", type=int, default=5, help="n fold")
    parser.add_argument("--randomSeed", type=int, help="randomSeed for reproduction")
    parser.add_argument("--preEmbd", default=False, action='store_true', help="calculate embedding before training")
    parser.add_argument("--dynamicSampling", default=False, action='store_true', help="sample based on class count")
    parser.add_argument("--model", default='clsTopic', help="model used for evaluation")
    parser.add_argument("--num_epoches", type=int, default=200, help="num epoches")
    parser.add_argument("--patient", type=int, default=4, help="early_stop_patient")
    parser.add_argument("--max_sent_len", type=int, default=300, help="maximum words in the sentence")
    parser.add_argument("--earlyStopping", default='cls_loss', help="early stopping")
    parser.add_argument("--corpusType", default='wvmisinfo', help="corpus type, for select reader")
    parser.add_argument("--splitValidation", type=float, help="split data from training for validation")
    parser.add_argument("--inspectTest", default=False, action='store_true', help="inspect testing data performance")
    parser.add_argument("--x_fields", help="x fileds", default='Claim,Explaination')
    parser.add_argument("--y_field", help="y filed", default='category')
    parser.add_argument("--trainOnly", help="only train the model, no split or test", default=False, action='store_true')
    parser.add_argument("--export_json", help="export json for scholar, need file path")
    parser.add_argument("--export_doc", help="export doc for npmi, need file path")
    parser.add_argument("--trainLDA", help="lda test", default=False, action='store_true')
    args = parser.parse_args()

    dictargs = vars(args)
    trainReaderargs = args.trainReaderargs.split(',')
    if args.testReaderargs:
        testReaderargs = args.testReaderargs.split(',')
    else:
        testReaderargs = None

    if args.valReaderargs:
        valReaderargs = args.valReaderargs.split(',')
    else:
        valReaderargs = None



    evaluaton = EvaluationManager(trainReaderargs, dictargs, testReaderargs=testReaderargs, valReaderargs=valReaderargs)

    if args.export_json:
        jsonData = evaluaton.get_covid_train_json_for_scholar()
        with open(args.export_json, 'w') as fo:
            json.dump(jsonData, fo)

    elif args.export_doc:
        all_doc = evaluaton.outputCorpus4NPMI()
        with open(args.export_doc, 'w') as fo:
            for item in all_doc:
                item = item.strip()
                fo.write(item+'\n')


    elif args.trainOnly:
        evaluaton.train_model_only()

    elif testReaderargs:
        evaluaton.train_test_evaluation()

    else:
        evaluaton.cross_fold_evaluation()





