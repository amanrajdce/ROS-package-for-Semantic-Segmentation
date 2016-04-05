import shelve   
import matplotlib.pyplot as pl
from path import Path
import numpy as np

avg_error = []
batches = []
f = shelve.open('test_metrics.shelf','r')

for keys in sorted(f.keys()):
    avg_error.append(f[keys]['average error'])
    batches.append(f[keys]['batch'])

f.close()
avg_accuracy = 1-np.mean(avg_error)

f = open("test-stats-VGG.txt",'w')
f.write("Pixel wise accuracy: %f \n" % (avg_accuracy*100))
f.write("Pixel wise error: %f" % (np.mean(avg_error)*100))
f.close()
print 'Pixel wise accuracy:', avg_accuracy*100
print 'Pixel wise error: ', np.mean(avg_error)*100

