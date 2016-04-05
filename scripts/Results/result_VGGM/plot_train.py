import shelve   
import matplotlib.pyplot as pl
from path import Path

#base_dir = Path('/home/aman/data/NYUv2_HHA/')

avg_error = []
avg_loss = []
epoch = []
test_error = []
#base_dir = Path('/home/aman/data/NYUv2_HHA/')
f = shelve.open('train_metrics_epoch.shelf','r')

for keys in sorted(f.keys()):
    avg_error.append(f[keys]['average error'])
    avg_loss.append(f[keys]['average loss'])
    epoch.append(f[keys]['epoch'])
    test_error.append(f[keys]['test_error'])
    #batches.append(f[keys]['batch'])

f.close()
'''
f = shelve.open('train_metrics_epocha54.shelf','r')
for keys in sorted(f.keys()):
    avg_error.append(f[keys]['average error'])
    avg_loss.append(f[keys]['average loss'])
    epoch.append(f[keys]['epoch'])

f.close()
'''
print 'epochs:', len(epoch)

pl.plot(xrange(len(epoch)), avg_loss); pl.ylabel('Losses'); pl.xlabel('Epoch');
pl.title('Training loss');
pl.savefig("loss_graph.png");
pl.close();

pl.plot(xrange(len(epoch)), avg_error); pl.ylabel('Error'); pl.xlabel('Epoch'); 
pl.title('Training error');
pl.savefig("error_graph.png")
pl.close();

pl.plot(xrange(len(epoch)), avg_error,label="train"); 
pl.plot(xrange(len(epoch)), test_error, label="test");
pl.legend(loc='upper right');
pl.ylabel('Error'); pl.xlabel('Epoch'); 
pl.title('Training and Testing error');
pl.savefig("train_test_graph.png")
pl.close();

'''
pl.plot(batches, avg_error); pl.ylabel('Error'); pl.xlabel('Batch');
pl.title('Testing error');
pl.savefig("test_error.png");
pl.close();
'''
