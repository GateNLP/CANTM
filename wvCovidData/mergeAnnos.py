import sys
from WvLibs import WVdataIter
import re
from itertools import combinations
import argparse
import json
import logging

global logger

class Wv_loger:
    def __init__(self, logging_level):
        self.logger = logging.getLogger('root')
        if logging_level == 'info':
            self.logger.setLevel(logging.INFO)
        elif logging_level == 'debug':
            self.logger.setLevel(logging.DEBUG)
        elif logging_level == 'warning':
            self.logger.setLevel(logging.WARNING)

        #self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('ROOT: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        #self.logger.info('info message')

    def logging(self, *args, logtype='debug', sep=' '):
        getattr(self.logger, logtype)(sep.join(str(a) for a in args))

    def info(self, *args, sep=' '):
        getattr(self.logger, 'info')(sep.join(str(a) for a in args))

    def debug(self, *args, sep=' '):
        getattr(self.logger, 'debug')(sep.join(str(a) for a in args))

    def warning(self, *args, sep=' '):
        getattr(self.logger, 'warning')(sep.join(str(a) for a in args))


def clean_string(text):
    text = text.strip()
    text = text.replace('\t','')
    text = text.replace('\n','')
    return text


def getList(dict):
    list = []
    for key in dict.keys():
        list.append(key)
    return list


def check_pare(ann1, ann2, agreeDict):
    combine1 = ann1+'|'+ann2
    combine2 = ann2+'|'+ann1
    combines = [combine1, combine2]
    combineInDict = False
    for combine in combines:
        if combine in agreeDict:
            combineInDict = True
            break
    return combine, combineInDict


def update_disagreement_dict(disagreeLabel1, disagreeLabel2, class_disagree_check_dict):
    if disagreeLabel1 not in class_disagree_check_dict:
        class_disagree_check_dict[disagreeLabel1] = {}
    if disagreeLabel2 not in class_disagree_check_dict[disagreeLabel1]:
        class_disagree_check_dict[disagreeLabel1][disagreeLabel2] = 0

    class_disagree_check_dict[disagreeLabel1][disagreeLabel2] += 1
    return class_disagree_check_dict

def calAgreement(dataIter, numlabels=11):
    compare_list = []
    pe = 1/numlabels
    
    ## get all paires
    for item in dataIter:
        tmp_list = []
        anno_list = []
        for annotation in item['annotations']:
            label = annotation['label']
            confident = annotation['confident']
            annotator = annotation['annotator']
            if annotator not in anno_list:
                anno_list.append(annotator)
                tmp_list.append([annotator, label])

            combine_list = list(combinations(tmp_list, 2))
            compare_list += combine_list

    t=0
    a=0
    agreeDict = {}

    for compare_pair in compare_list:
        ann1 = compare_pair[0][0]
        label1 = compare_pair[0][1]
        ann2 = compare_pair[1][0]
        label2 = compare_pair[1][1]
        t+=1
        combine, combineInDict = check_pare(ann1, ann2, agreeDict)

        if combineInDict:
            agreeDict[combine]['t'] += 1
        else:
            agreeDict[combine] = {}
            agreeDict[combine]['t'] = 1
            agreeDict[combine]['a'] = 0
            agreeDict[combine]['disagree'] = {}
            agreeDict[combine]['disagree'][ann1] = []
            agreeDict[combine]['disagree'][ann2] = []

        if label1 == label2:
            a+=1
            agreeDict[combine]['a'] += 1
        else:
            agreeDict[combine]['disagree'][ann1].append(label1)
            agreeDict[combine]['disagree'][ann2].append(label2)



    pa = a/t
    #pe = 1/numlabels
    #print(pe)
    overall_kappa = (pa-pe)/(1-pe)
    logger.warning('overall agreement: ', pa)
    logger.warning('overall kappa: ', overall_kappa)
    logger.warning('total pair compareed: ', t)
    logger.warning('annotator pair agreement kappa num_compared')
    class_disagree_check_dict = {}

    for annPair in agreeDict:
        logger.info('\n')
        logger.info('============')
        num_compared = agreeDict[annPair]['t']
        cpa = agreeDict[annPair]['a']/agreeDict[annPair]['t']
        kappa = (cpa-pe)/(1-pe)
        logger.info(annPair, cpa, kappa, num_compared)
        keys = getList(agreeDict[annPair]['disagree'])
        logger.info(keys)
        for i in range(len(agreeDict[annPair]['disagree'][keys[0]])):
            disagreeLabel1 = agreeDict[annPair]['disagree'][keys[0]][i]
            disagreeLabel2 = agreeDict[annPair]['disagree'][keys[1]][i]
            logger.info(disagreeLabel1, disagreeLabel2)
            class_disagree_check_dict = update_disagreement_dict(disagreeLabel1, disagreeLabel2, class_disagree_check_dict)
            class_disagree_check_dict = update_disagreement_dict(disagreeLabel2, disagreeLabel1, class_disagree_check_dict)

    logger.info('\n')
    logger.info('=========================')
    logger.info('class level disagreement')
    logger.info('=========================')
    for item_label in class_disagree_check_dict:
        logging.info(item_label)
        logging.info(class_disagree_check_dict[item_label])
        logging.info('===================')
        logging.info('\n')
    return pa, overall_kappa







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_json_dir", help="Unannotated Json file dir")
    parser.add_argument("annoed_json_dir", help="Annotated Json file dir")
    parser.add_argument("merged_json", help="merged json")
    parser.add_argument("--ignoreLabel", help="ignore label, splited by using ,")
    parser.add_argument("--ignoreUser", help="ignore user, splited by using ,")
    parser.add_argument("--min_anno_filter", type=int, default=1, help="min annotation frequence")
    parser.add_argument("--min_conf_filter", type=int, default=-1, help="min confident")
    parser.add_argument("--output2csv", help="output to csv")
    parser.add_argument("--transfer_label", default=None, help="trasfer label to another category, in format: orilabel1:tranlabel1,orilabel2:tranlabel2")
    parser.add_argument("--cal_agreement", action='store_true',help="calculate annotation agreement")
    parser.add_argument("--logging_level", default='warning', help="logging level, default warning, other option inlcude info and debug")
    parser.add_argument("--user_conf", default=None, help="User level confident cutoff threshold, in format: user1:thres1,user2:thres2")
    parser.add_argument("--set_reverse", default=False, action='store_true', help="reverse the selection condition, to check what discared")

    #parser.add_argument("--logging_file", default='default.log',help="logging file")


    args = parser.parse_args()

    #logger = logging.getLogger()

    output2csv = None
    trans_dict = {}



    raw_json_dir = args.raw_json_dir
    annoed_json_dir = args.annoed_json_dir
    merged_json = args.merged_json
    min_frequence = args.min_anno_filter
    min_conf = args.min_conf_filter

    list2ignor = []
    user2ignor = []
    if args.ignoreLabel:
        list2ignor = args.ignoreLabel.split(',')

    if args.ignoreUser:
        user2ignor = args.ignoreUser.split(',')

    logging_level = args.logging_level
    logger = Wv_loger(logging_level)
    



    if args.output2csv:
        output2csv = args.output2csv


    if args.transfer_label:
        all_trans_labels = args.transfer_label.split(',')
        for label_pair in all_trans_labels:
            tokened_pair = label_pair.split(':')
            trans_dict[tokened_pair[0]] = tokened_pair[1]
    
    user_conf_dict = {}
    if args.user_conf:
        all_user_conf = args.user_conf.split(',')
        for conf_pair in all_user_conf:
            tokened_pair = conf_pair.split(':')
            user_conf_dict[tokened_pair[0]] = int(tokened_pair[1])

    logger.warning(trans_dict)


    dataIter = WVdataIter(annoed_json_dir, raw_json_dir, min_anno_filter=min_frequence, ignoreLabelList=list2ignor, ignoreUserList=user2ignor, label_trans_dict=trans_dict, check_validation=False, confThres=min_conf, reverse_selection_condition=args.set_reverse, logging_level=logging_level, annotator_level_confident=user_conf_dict)
    data2merge = dataIter.getMergedData()
    print('num selected data:', len(data2merge))
    with open(merged_json, 'w') as fj:
        json.dump(data2merge, fj)

    num_labels = 11


    if output2csv:
        t=0
        num_anno_dict = {}
        num_label_dict = {}
        with open(output2csv, 'w') as fo:
            outline = 'id\tclaim\texplaination\tselected_label\tlables_from_annotator\n'
            fo.write(outline)
            for item in data2merge:
                t += 1
                num_annotation = len(item['annotations'])
                if num_annotation not in num_anno_dict:
                    num_anno_dict[num_annotation] = 0
                num_anno_dict[num_annotation] += 1


                claim = clean_string(item['Claim'])
                explaination = clean_string(item['Explaination'])
                unique_id = item['unique_wv_id'].strip()
                if 'selected_label' in item:
                    selected_label = item['selected_label']
                else:
                    selected_label = ''
                outline = unique_id+'\t'+claim+'\t'+explaination+'\t'+selected_label
                for annotation in item['annotations']:
                    label = annotation['label']
                    confident = annotation['confident']
                    annotator = annotation['annotator']
                    if label not in num_label_dict:
                        num_label_dict[label] = 0
                    num_label_dict[label] += 1

                    outline += '\t'+label+'\t'+confident+'\t'+annotator
                outline += '\n'
                fo.write(outline)
    print(num_anno_dict)
    print(num_label_dict)
    num_labels = len(num_label_dict)



    if args.cal_agreement:
        pa, kappa = calAgreement(dataIter, num_labels)
        print('agreement: ', pa)
        print('kappa: ', kappa)

