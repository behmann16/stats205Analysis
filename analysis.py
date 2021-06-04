import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

#change this to "VALprocessed.csv" for valorant data
train = pd.read_csv("CSprocessed.csv").values

def gauss(val):
    coeff = (2 * 3.14) ** (-1/2)
    exp = -1/2 * ((val) ** 2)

    return coeff * (2.718 ** (exp))


def gaussian(data):

    h = 5

    ret = []
    for line in data:
        #line[1] for util, line[0] for aim
        x = float(line[0])
        y = float(line[2])
        sum = 0.0
        n = 0
        for nline in data:
            sum += float(nline[1]) * gauss((float(nline[0]) - x) / h)
            n += gauss((float(nline[0]) - x) / h)
        avg = 0.0
        if n > 0:
            avg = sum / n
        ret.append([x, avg])

    return ret



gs = gaussian(train)


plt.plot([row[0] for row in gs], [row[1] for row in gs], label='gke')


test = pd.read_csv("CStest.csv").values

correct = 0
total = 0
for match in test:

    #match[1] for util
    predict = gs[match[0]]
    if predict >= 0.5:
        predict = 1
    else:
        predict = 0

    if predict == match[2]:
        correct += 1
    total += 1
accuracy = float(correct  / total)

print("accuracy: ", accuracy)

