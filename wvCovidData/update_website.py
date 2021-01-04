import json
import re
import nltk
import sys
import string
import copy


class MediaTypeOrg:
    def __init__(self):
        punct_chars = list(set(string.punctuation)-set("_"))
        print(punct_chars)
        punctuation = ''.join(punct_chars)
        self.replace = re.compile('[%s]' % re.escape(punctuation))

        self.multi_word_dict = {'social media':'social_media'}


        self.facebook = ['facebook', 'fb', 'faceboos', 'facebok', 'faceboook', 'facebooks', 'facebbok','faacebook','facebookk']
        self.twitter = ['twitter', 'tweets','tweet']
        self.news = ['media', 'news', 'newspaper', 'newspapers', 'times', 'abcnews','cgtn']
        self.whatsApp = ['whatsapp', 'wa']
        self.email = ['email']
        self.social_media = ['social_media', 'weibo', 'wechat'] #######
        self.youtube = ['youtube', 'youtuber']
        self.blog = ['blog', 'bloggers', 'blogs', 'blogger']
        self.instagram = ['instagram', 'ig']
        self.tv = ['tv', 'television']
        self.line = ['line']
        self.tiktok = ['tiktok']
        self.chainMessage = ['chain', 'telegram']

        self.type_dict={
                'facebook': self.facebook,
                'twitter': self.twitter,
                'news': self.news,
                'whatsapp': self.whatsApp,
                'email': self.email,
                'youtube': self.youtube,
                'blog': self.blog,
                'instagram': self.instagram,
                'tv': self.tv,
                'line': self.line,
                'chain_message': self.chainMessage,
                'other_social_media': self.social_media,
                'social_media': self.social_media+self.facebook+self.twitter+self.instagram,
                'tiktok': self.tiktok
                }


    def type_org(self, data):
        addi_type_dict = {}
        ori_web_claim_dict = {}
        num_other = 0

        new_data = []

        for each_data in data:
            website_ori = each_data['Claim_web_ori']
            included_type_list, tokened = self.get_type(website_ori)
            if len(included_type_list) == 0:
                num_other += 1
                print(website_ori)
                included_type_list.append('other')
            each_data['Claim_Website'] = included_type_list
            new_data.append(copy.deepcopy(each_data))
        return new_data

    def _check_item_in_list(self, tokened, check_list):
        for item in tokened:
            if item in check_list:
                return True
        return False
    
    def _get_multiword(self, lower_case_string):
        for item in self.multi_word_dict:
            lower_case_string = re.sub(item, self.multi_word_dict[item], lower_case_string)
        return lower_case_string



    def get_type(self, tokened):
        #print(tokened)
        #print(raw_media_type)
        #lc_raw_mt = raw_media_type.lower()
        #lc_raw_mt = self._get_multiword(lc_raw_mt)
        #lc_raw_mt_re = self.replace.sub(' ', lc_raw_mt)

        #tokened = nltk.word_tokenize(lc_raw_mt_re)


        included_type_list = []
        for media_type in  self.type_dict:
            check_list = self.type_dict[media_type]
            type_included = self._check_item_in_list(tokened, check_list)
            if type_included:
                included_type_list.append(media_type)
        ###check media
        #for search_string in self.social_media:
        #    m = re.search(search_string, lc_raw_mt)
        #    if m:
        #        #print(raw_media_type)
        #        included_type_list.append('social_media')

        #print(included_type_list)

        return included_type_list, tokened

            
json_file = sys.argv[1]
output_file = sys.argv[2]

with open(json_file, 'r') as fj:
    data = json.load(fj)


typeCheck = MediaTypeOrg()
new_data = typeCheck.type_org(data)

with open(output_file, 'w') as fo:
    json.dump(new_data, fo)



