from .DataReader import DataReader
import random

def disagreementSolver(annotations, local_logger=None):
    mergeMethodList = ['single_annotation', 'all_agreement', 'majority_agreement', 'majority_confidence', 'highest_confidence']
    mergeMethod = None
    annotation_list = []
    label_list = []
    annotator_list = []
    all_labels_include_duplicats = []
    for annotation in annotations:
        label = annotation['label']
        confident = annotation['confident']
        annotator = annotation['annotator']
        all_labels_include_duplicats.append([label, confident, annotator])
        # for each annotator, we only consider once
        if annotator not in annotator_list:
            annotator_list.append(annotator)
            annotation_list.append([label, confident, annotator])
            label_list.append(label)

    label_set = list(set(label_list))
    set_count = [0]*len(label_set)
    conf_sum = [0]*len(label_set)

    for idx, current_label in enumerate(label_set):
        set_count[idx] = label_list.count(current_label)
    sorted_count = sorted(set_count, reverse=True)

    for each_annotation in annotation_list:
        label_set_idx = label_set.index(each_annotation[0])
        try:
            conf_sum[label_set_idx] += int(each_annotation[1])
        except:
            pass
    sorted_conf_sum = sorted(conf_sum, reverse=True)

    ## if only 1 label, or all annotator agree eachother
    if len(label_set) == 1:
        selected_label = label_set[0]
        if len(annotation_list) == 1:
            mergeMethod = 'single_annotation'
        else:
            mergeMethod = 'all_agreement'
            
    ## if have majority agreement    
    elif sorted_count[0] > sorted_count[1]:
        selected_label_idx = set_count.index(sorted_count[0])
        selected_label = label_set[selected_label_idx]
        mergeMethod = 'majority_agreement'

    ## if have higer summed confidence
    elif sorted_conf_sum[0] > sorted_conf_sum[1]:
        selected_label_idx = conf_sum.index(sorted_conf_sum[0])
        selected_label = label_set[selected_label_idx]
        mergeMethod = 'majority_confidence'
    ## else pick the lable have highest confidence
    else:
        sorted_annotation_list = sorted(annotation_list, key=lambda s:s[1], reverse=True)
        selected_label = sorted_annotation_list[0][0]
        mergeMethod = 'highest_confidence'

    if local_logger:
        local_logger(selected_label, annotation_list, logtype='debug')
    return selected_label, mergeMethod


class WVdataIter(DataReader):
    def __init__(self, annoed_json_dir, raw_json, min_anno_filter=-1, postProcessor=None, shuffle=False, check_validation=False, **kwargs):
        super().__init__(annoed_json_dir, raw_json, kwargs)#ignoreLabelList=ignoreLabelList, ignoreUserList=ignoreUserList, confThres=confThres, ignore_empty=ignore_empty)
        self.shuffle = shuffle
        self.check_validation = check_validation
        self.filterByMinAnno(min_anno_filter)
        self._reset_iter()
        self.postProcessor = postProcessor


    def filterByMinAnno(self, min_anno_filter):
        self.min_anno_filter = min_anno_filter
        all_links = []
        for link in self.data_dict:
            num_annotations = len(self.data_dict[link]['annotations'])
            if num_annotations >= self.min_anno_filter:
                if self.check_validation:
                    if self._check_annotation_valid(self.data_dict[link]['annotations']):
                        all_links.append(link)
                else:
                    all_links.append(link)

        self.all_links = all_links


    def _check_annotation_valid(self, annotation):
        at_least_one_ture = False
        for item in annotation:
            current_label = item['label']
            current_confident = item['confident']
            if len(current_label) > 0 and len(current_confident)>0:
                at_least_one_ture = True
        return at_least_one_ture

        
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.all_links)

        self._reset_iter()
        return self

    def __next__(self):
        if self.current_sample_idx < len(self.all_links):
            current_sample = self._readNextSample()
            self.current_sample_idx += 1
            if self.postProcessor:
                return self.postProcessor(current_sample)
            else:
                return current_sample

        else:
            self._reset_iter()
            raise StopIteration


    def _readNextSample(self):
        current_link = self.all_links[self.current_sample_idx]
        current_sample = self.data_dict[current_link]
        return current_sample


    def __len__(self):
        return len(self.all_links)

    def _reset_iter(self):
        self.current_sample_idx = 0


    def getMergedData(self, disagreementSolver=disagreementSolver):
        merge_method_dict = {}
        merged_data_list = []
        merged_label_count = {}
        for item in self:
            if (len(item['annotations']) > 0) and disagreementSolver:
                annotations = item['annotations']
                selected_label,merge_method = disagreementSolver(annotations, self.logging)
                if merge_method not in merge_method_dict:
                    merge_method_dict[merge_method] = 0
                if selected_label not in merged_label_count:
                    merged_label_count[selected_label] = 0
                item['selected_label'] = selected_label
                merge_method_dict[merge_method] += 1
                merged_label_count[selected_label] += 1
                merged_data_list.append(item)
        self.logging(merge_method_dict, logtype='info')
        self.logging(merged_label_count, logtype='info')
        return merged_data_list
