import numpy as np

ALPHABET = 'abcdefg123'
N_ALPHABET = len(ALPHABET)
list_char = [c for c in ALPHABET]
dict_char = {list_char[i]:i for i in range(len(list_char))}

def onehot(_num,_range):
    r  = [0 for i in range(_range)]
    r[_num] = 1
    return r 

def str2onehot(string):
    arr = []
    for c in string:
        arr.append(dict_char[c])
    res = []
    for e in arr:
        res.extend(onehot(e,N_ALPHABET))
    return res 

