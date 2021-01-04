import json
import re
import nltk
import argparse
import copy
import os
import sys
script_path = os.path.abspath(__file__)
print(script_path)
global parent
parent = os.path.dirname(script_path)
print(parent)
gatePython = os.path.join(parent, 'GATEpythonInterface')
sys.path.append(gatePython)
from GateInterface import *
import string
from pathlib import Path

source_out_folder = os.path.join(parent, 'test_text')
Path(source_out_folder).mkdir(parents=True, exist_ok=True)

printable = set(string.printable)

ignore_explist=[
        ['please', 'click', 'the', 'link', 'to', 'read', 'the', 'full', 'article'],
        ['please', 'read', 'the', 'full', 'article'],
        ]

def check_ori_claim_type(claim_web_ori):
    media_type = set()
    image = ['image', 'photo', 'picture', 'images', 'pictures', 'photos', 'infograph']
    video = ['video', 'videos', 'television']
    audio = ['audio', 'radio']
    text = ['text', 'articles', 'article','message', 'messages']

    #print(claim_web_ori)
    for each_tok in claim_web_ori:
        if each_tok in image:
            media_type.add('image')
        if each_tok in video:
            media_type.add('video')
        if each_tok in audio:
            media_type.add('audio')
        if each_tok in text:
            media_type.add('text')
    return list(media_type)

class JapeCheck:
    def __init__(self):
        self.gate = GateInterFace()
        self.gate.init()
        self.checkPipeline = GatePipeline('checkPipeline')
        self.checkPipeline.loadPipelineFromFile(os.path.join(parent, 'japeCheck.xgapp'))

    def check_jape(self, doc_name, claim_website):
        print(doc_name)
        document = GateDocument()
        document.loadDocumentFromFile(doc_name)
        #print(document)
        testCorpus = GateCorpus('testCorpus')
        testCorpus.addDocument(document)
        self.checkPipeline.setCorpus(testCorpus)
        self.checkPipeline.runPipeline()
        ats = document.getAnnotations('')
        media_type = ats.getType('mediaType')
        #print(media_type.annotationSet)
        media_type_dict = []
        for item in media_type:
            #print(item.features)
            media_type_dict.append(item.features)
        mediaTypeMatchedList, mediaTypeUnMatchedList = self.match_type(media_type_dict, claim_website)

        loose_media_dict = []
        loose_media = ats.getType('looseMedia')
        for item in loose_media:
            #print(item.features)
            loose_media_dict.append(item.features)

        looseMatchedList, looseUnMatchedList = self.match_type(loose_media_dict, claim_website)

        document.clearDocument()
        testCorpus.clearCorpus()

        return self.select_type(mediaTypeMatchedList, mediaTypeUnMatchedList, looseMatchedList, looseUnMatchedList)



    def count_match(self, matchedList):
        current_max_type = None
        current_max_count = 0
        all_possible_types = set(matchedList)
        for possible_type in all_possible_types:
            current_possible_count = matchedList.count(possible_type)
            if current_possible_count > current_max_count:
                current_max_type = possible_type
                current_max_count = current_possible_count
            elif current_possible_count == current_max_count:
                if possible_type == 'video':
                    current_max_type == 'video'
        source_media = [current_max_type]
        return source_media


    def select_type(self, mediaTypeMatchedList, mediaTypeUnMatchedList, looseMatchedList, looseUnMatchedList):
        source_media = []
        if len(mediaTypeMatchedList) > 0:
            source_media = mediaTypeMatchedList
        elif len(looseMatchedList) > 0:
            source_media = self.count_match(looseMatchedList)
            #source_media = [looseMatchedList[0]]
        elif len(mediaTypeUnMatchedList) > 0:
            source_media = mediaTypeUnMatchedList
        elif len(looseUnMatchedList) > 0:
            source_media = self.count_match(looseUnMatchedList)
            #current_max_type = None
            #current_max_count = 0
            #all_possible_types = set(looseUnMatchedList)
            #for possible_type in all_possible_types:
            #    current_possible_count = looseUnMatchedList.count(possible_type)
            #    if current_possible_count > current_max_count:
            #        current_max_type = possible_type
            #        current_max_count = current_possible_count
            #source_media = [current_max_type]
        return source_media


    def match_type(self,media_type_dict, webSite_list):
        matched_list = []
        unmatched_list = []
        for item in media_type_dict:
            try:
                current_web = item['oriWeb']
            except:
                current_web = 'unknown'
            try:
                current_media = item['mediaType']
            except:
                current_media = 'unknown_web'
            #print(current_web, current_media)
            current_web_list = current_web.split(',')
            #print(current_web_list)
            current_media_list_raw = current_media.split(',')
            current_media_list = []
            for media_item in current_media_list_raw:
                if len(media_item) > 0:
                    current_media_list.append(media_item)

            web_match = False
            for current_web_item in current_web_list:
                if current_web_item in webSite_list:
                    #matched_list.append(current_media)
                    web_match = True
                    break
                    #unmatched_list.append(current_media)
            if web_match:
                matched_list += current_media_list
            else:
                unmatched_list += current_media_list



        return matched_list, unmatched_list


parser = argparse.ArgumentParser()
parser.add_argument("inputJson", help="args for test reader")
parser.add_argument("outputJson", help="args for test reader")
parser.add_argument("--updateOnly", help="reverse selection criteria", default=False, action='store_true')
args = parser.parse_args()

input_json = args.inputJson
output_json = args.outputJson

with open(input_json, 'r') as fj:
    raw_data = json.load(fj)


japeCheck = JapeCheck()
num_video_source = 0
doc_id = 0
new_data = []
punct_chars = list(set(string.punctuation)-set("_"))
punctuation = ''.join(punct_chars)
pun_replace = re.compile('[%s]' % re.escape(punctuation))
for item in raw_data:
    update = True
    type_of_media = item['Type_of_media']
    claim = item['Claim']
    explanation = item['Explaination']

    claim = claim.lower()
    claim = pun_replace.sub(' ', claim)
    claim_tok = nltk.word_tokenize(claim)

    explanation = explanation.lower()
    explanation = pun_replace.sub(' ', explanation)
    explanation_tok = nltk.word_tokenize(explanation)


    #print(type_of_media)
    if args.updateOnly:
        if len(type_of_media) > 0:
            update = False
        else:
            print(doc_id)
            update = True
    #print(update)
    if update:
        item_keys = item.keys()
        #print(item_keys)
        claim_website = item['Claim_Website']
        #print(claim_website)
        claim_website_ori = item['Claim_web_ori']

        if 'p_tag' in item:
            text_source = item['p_tag']
        elif 'Source_PageTextEnglish' in item:
            text_source = item['Source_PageTextEnglish']
        else:
            #print(doc_id)
            text_source = item['Source_PageTextOriginal']
        
        source_media = []
        if ('youtube' in claim_website) or ('tv' in claim_website) or ('tiktok' in claim_website):
            num_video_source += 1
            source_media = ['video']

        else:
            media_type = check_ori_claim_type(claim_website_ori)
            claim_media_type = check_ori_claim_type(claim_tok)
            if explanation_tok not in ignore_explist:
                exp_media_type = check_ori_claim_type(explanation_tok)
            else:
                print('!!!!!!!!', explanation_tok)
                exp_media_type = []
            if len(media_type) > 0:
                num_video_source += 1
                source_media = media_type
            elif len(claim_media_type) > 0:
                num_video_source += 1
                source_media = claim_media_type
                #print(source_media, claim_tok)
            elif len(exp_media_type) > 0:
                num_video_source += 1
                source_media = exp_media_type
                print(source_media, explanation_tok)

            else:
                print(claim_website_ori)
                text_out = os.path.join(parent, 'test_text')
                text_out = os.path.join(text_out, str(doc_id)+'.txt')
                filterd_text_source = ''.join(filter(lambda x: x in printable, text_source))
                #print(filterd_text_source)
                if len(filterd_text_source) > 10:
                    with open(text_out, 'w') as fo:
                        fo.write(filterd_text_source)
                    source_media = japeCheck.check_jape(text_out, claim_website)
                    num_video_source += 1
        doc_id += 1
        print(source_media)
        item['Type_of_media'] = list(set(source_media))
    new_data.append(copy.deepcopy(item))


print(num_video_source)

with open(output_json, 'w') as fo:
    json.dump(new_data, fo)



