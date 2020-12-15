import math
import random


class BatchIterBert:
    def __init__(self, dataIter, batch_size=32, filling_last_batch=False, postProcessor=None):
        self.dataIter = dataIter
        self.batch_size = batch_size
        self.num_batches = self._get_num_batches()
        self.filling_last_batch = filling_last_batch
        self.postProcessor = postProcessor
        self.fillter = []
        self._reset_iter()

    def _get_num_batches(self):
        num_batches = math.ceil(len(self.dataIter)/self.batch_size)
        return num_batches

    def _reset_iter(self):
        self.current_batch_idx = 0

    def __iter__(self):
        self._reset_iter()
        return self

    def __next__(self):
        if self.current_batch_idx < self.num_batches:
            current_batch_x, current_batch_y = self._readNextBatch()
            self.current_batch_idx += 1
            if self.postProcessor:
                return self.postProcessor(current_batch_x, current_batch_y)
            else:
                return current_batch_x, current_batch_y

        else:
            self._reset_iter()
            raise StopIteration
    def __len__(self):
        return self.num_batches

    def _readNextBatch(self):
        i = 0
        batch_list_x = []
        batch_list_y = []
        while i < self.batch_size:
            try:
                x, y = next(self.dataIter)
                if self.filling_last_batch:
                    self._update_fillter(x, y)
                batch_list_x.append(x)
                batch_list_y.append(y)
                i+=1
            except StopIteration:
                if self.filling_last_batch:
                    batch_list_x, batch_list_y = self._filling_last_batch(batch_list_x, batch_list_y)
                i = self.batch_size
        return batch_list_x, batch_list_y


    def _filling_last_batch(self, batch_list_x, batch_list_y):
        num_current_batch = len(batch_list_x)
        num_filling = self.batch_size - num_current_batch
        random.shuffle(self.fillter)
        filler_x = [s[0] for s in self.fillter[:num_filling]]
        filler_y = [s[1] for s in self.fillter[:num_filling]]
        batch_list_x += filler_x
        batch_list_y += filler_y
        return batch_list_x, batch_list_y

    def _update_fillter(self, x, y):
        r = random.random()
        if len(self.fillter) < self.batch_size:
            self.fillter.append([x, y])
        elif r>0.9:
            self.fillter.pop(0)
            self.fillter.append([x, y])

