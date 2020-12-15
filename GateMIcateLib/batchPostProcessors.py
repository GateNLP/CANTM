import torch
def xonlyBatchProcessor(x, y):
    ss = [s[1] for s in x]
    return ss[0]


def bowBertBatchProcessor(raw_x, y):
    x = [s[0] for s in raw_x]
    idded_words = [s[1] for s in raw_x]

    y_class = y
    return torch.tensor(x), torch.tensor(idded_words), torch.tensor(y_class)

def xyOnlyBertBatchProcessor(raw_x, y):
    x = [s[0] for s in raw_x]
    y_class = y
    return torch.tensor(x), torch.tensor(y_class)

def singleProcessor_noy(raw_x):
    x = [raw_x[0]]
    return torch.tensor(x)

