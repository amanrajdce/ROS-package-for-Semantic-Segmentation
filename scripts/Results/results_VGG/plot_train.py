import shelve   
import matplotlib.pyplot as pl
from path import Path

#base_dir = Path('/home/aman/data/NYUv2_HHA/')

avg_error = []
avg_loss = []
epoch = []
#batches = []
f = shelve.open('train_metrics_epoch108.shelf','r')

for keys in sorted(f.keys()):
    avg_error.append(f[keys]['average error'])
    avg_loss.append(f[keys]['average loss'])
    epoch.append(f[keys]['epoch'])
    #batches.append(f[keys]['batch'])

f.close()

f = shelve.open('train_metrics_epocha108.shelf','r')
for keys in sorted(f.keys()):
    avg_error.append(f[keys]['average error'])
    avg_loss.append(f[keys]['average loss'])
    epoch.append(f[keys]['epoch'])

f.close()

print 'epochs:', len(epoch)

pl.plot(xrange(len(epoch)), avg_loss); pl.ylabel('Losses'); pl.xlabel('Epoch');
pl.title('Training loss');
pl.savefig("loss_graph.png");
pl.close();

pl.plot(xrange(len(epoch)), avg_error); pl.ylabel('Error'); pl.xlabel('Epoch'); 
pl.title('Training error');
pl.savefig("error_graph.png")
pl.close();
'''
pl.plot(batches, avg_error); pl.ylabel('Error'); pl.xlabel('Batch');
pl.title('Testing error');
pl.savefig("test_error.png");
pl.close();
'''
