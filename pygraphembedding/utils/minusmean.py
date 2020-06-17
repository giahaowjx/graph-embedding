import numpy as np

# 传入一些列图片
def minusmean(data):
    if len(data) == 0:
        return

    mean = data.mean(axis=0)
    return data - mean
