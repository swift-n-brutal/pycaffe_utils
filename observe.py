import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import time
import copy

import caffe
import lmdb

mean_rgb = np.array([125.3, 123.0, 113.9])
std_rgb = np.array([63.0, 62.1, 66.7])
n_images = 50000
ROOT = "../../"
DEPLOY_PATH = osp.join(ROOT, "examples/cifar10/resnet20_cifar10_1st_deploy.prototxt")
MODEL_PATH = osp.join(ROOT, "models/resnet_cifar10/resnet20_cifar10_1st_bz128_B_iter_32000.caffemodel")

def init_net(deploy=None, model=None, phase=caffe.TRAIN):
    caffe.set_device(1)
    caffe.set_mode_gpu()
    
    if deploy == None:
        deploy = DEPLOY_PATH
    if model == None:
        model = MODEL_PATH
    net = caffe.Net(deploy, model, phase)
    return net

def compute_meanstd(txn):
    cursor = txn.cursor()
    n_images = 0
    mean_rgb = np.zeros(3)
    std_rgb = np.zeros(3)
    for _, raw_datum in cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(raw_datum)
        flat_x = np.fromstring(datum.data, dtype=np.uint8)
        x = flat_x.reshape(datum.channels, -1)
        mean_rgb += np.mean(x, 1)
        n_images += 1
    mean_rgb /= n_images
    
    for _, raw_datum in cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(raw_datum)
        flat_x = np.fromstring(datum.data, dtype=np.uint8)
        x = flat_x.reshape(datum.channels, -1) - mean_rgb.reshape(datum.channels, -1)
        std_rgb += np.mean(np.square(x), 1)
    std_rgb /= n_images
    std_rgb = np.sqrt(std_rgb)
    return mean_rgb, std_rgb
    
def init_loader(path):
    env = lmdb.open(path, readonly=True)
    txn = env.begin()
    return txn

def load_image(txn, index, key_length=5):
    raw_datum = txn.get(eval("'%%0%dd' %% index" % key_length))
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)
    flat_x = np.fromstring(datum.data, dtype=np.uint8)
    x = flat_x.reshape(datum.channels, datum.height, datum.width)
    y = datum.label
    return x, y

def load_batch(txn, batch, dest_x, dest_y):
    for i,index in enumerate(batch):
        x,y = load_image(txn, index)
        dest_x[i,...] = x
        dest_y[i] = y
    
def load_dataset(path="cifar10_train_lmdb", recompute=False):
    txn = init_loader(path)
    # get the shape
    x,y = load_image(txn, 0)
    c,h,w = x.shape
    if recompute:
        mean_rgb, std_rgb = compute_meanstd(txn)
        print "Mean", mean_rgb
        print "Std", std_rgb
    data = np.zeros((n_images, c, h, w))
    label = np.zeros((n_images))
    batch = np.arange(n_images)
    load_batch(txn, batch, data, label)
    dataset = {"data": data,
               "label": label}
    return dataset

def load_batch_from_dataset(data, label, batchid, dest_x, dest_y):
    for i,index in enumerate(batchid):
        dest_x[i,...] = data[index,...]
        dest_y[i] = label[index]

def sample_batch(batchsize):
    return np.random.randint(n_images, size=batchsize)

def fill_input(net, batchid, data, label, transform=True):
    load_batch_from_dataset(data, label, batchid, net.blobs['data'].data, net.blobs['label'].data)
    if transform:
        net.blobs['data'].data[...] -= mean_rgb.reshape(1,3,1,1)
        net.blobs['data'].data[...] /= std_rgb.reshape(1,3,1,1)

def update_net(net, history, lr=0.1, mom=0.9, decay=0.0001):
    for i, l in enumerate(net.layers):
        if l.type == "Convolution":
            history[i][0][...] = history[i][0]*mom + l.blobs[0].diff \
                + l.blobs[0].data*decay
            l.blobs[0].data[...] -= lr*history[i][0]
        elif (l.type == "Scale") or (l.type == "InnerProduct"):
            history[i][0][...] = history[i][0]*mom + l.blobs[0].diff \
                + l.blobs[0].data*decay
            l.blobs[0].data[...] -= lr*history[i][0]
            history[i][1][...] = history[i][1]*mom + l.blobs[1].diff \
                + l.blobs[1].data*decay
            l.blobs[1].data[...] -= lr*history[i][1]
            
def init_history_net(net):
    history = list()
    for l in net.layers:
        if l.type == "Convolution":
            history.append([np.zeros_like(l.blobs[0].data)])
        elif (l.type == "Scale") or (l.type == "InnerProduct"):
            history.append([np.zeros_like(l.blobs[0].data),
                            np.zeros_like(l.blobs[1].data)])
        else:
            history.append(None)
    return history

def clear_diff_net(net):
    for l in net.layers:
        if l.type == "Convolution":
            l.blobs[0].diff[...] = 0.
        elif (l.type == "Scale") or (l.type == "InnerProduct"):
            l.blobs[0].diff[...] = 0.
            l.blobs[1].diff[...] = 0.

def step_net(net, history, dataset, lr=0.1, mom=0.9, decay=0.0001, verbose=False):
    batchsize_in = net.blobs['data'].num
    batchid_in = sample_batch(batchsize_in)
    fill_input(net, batchid_in, dataset['data'], dataset['label'])
    output_fw_in = net.forward()
    # important !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    clear_diff_net(net)
    # ---------------------------------------------------
    output_bw_in = net.backward()
    update_net(net, history, lr, mom, decay)
    if verbose:
        print output_fw_in
    return output_fw_in, output_bw_in, batchid_in

def step_n_net(net, history, dataset, n, lr=0.1, mom=0.9, decay=0.0001, verbose=False):
    avg = None
    for i in xrange(n):
        ofw, _, _ = step_net(net, history, dataset, lr, mom, decay, verbose)
        if avg == None:
            avg = dict()
            for k,v in ofw.items():
                avg[k] = np.copy(v)
        else:
            for k,v in ofw.items():
                avg[k] += v
    for k in avg.keys():
        avg[k] /= n
    return avg
                

def get_blobs_act(net):
    act = list()
    for i,l in enumerate(net.layers):
        if l.type == "ReLU":
            blob_ids = net._top_ids(i)
            blob_name = net._blob_names[blob_ids[0]]
            act.append(np.copy(net.blobs[blob_name].data))
    return act

def get_blobs_sgn_from_act(act):
    sgn = list()
    for a in act:
        sgn.append(a != 0)
    return sgn
    
def get_blobs_sgn(net):
    act = get_blobs_act(net)
    return get_blobs_sgn_from_act(act)

def get_trans(sgn0, sgn1):
    return np.sum(np.logical_xor(sgn0,sgn1))
    
def get_trans_pos(sgn0, sgn1):
    return np.sum(np.logical_and(np.negative(sgn0), sgn1))
        
def get_trans_neg(sgn0, sgn1):
    return np.sum(np.logical_and(sgn0, np.negative(sgn1)))

def init_all(deploy=None, model=None, phase=caffe.TRAIN,
             path="cifar10_train_lmdb", recompute=False):
    net = init_net(deploy, model, phase)
    history = init_history_net(net)
    dataset = load_dataset(path, recompute)
    return net, history, dataset

def plot_trans(trans_before, trans_after):
    N = trans_before.shape[1]
    tb_mean = np.mean(trans_before, axis=0)
    tb_std = np.std(trans_before, axis=0)
    
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, tb_mean, width, color='r', yerr=tb_std)
    
    ta_mean = np.mean(trans_after, axis=0)
    ta_std = np.mean(trans_after, axis=0)
    rects2 = ax.bar(ind + width, ta_mean, width, color='y', yerr=ta_std)
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Number of sign transitions')
    ax.set_title('Change of sign transitions before/after dropping learning rate')
    ax.set_xlabel('Layer')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(np.arange(1,N+1))
    
    ax.legend((rects1[0], rects2[0]), ('Before dropping', 'After dropping'))
    
    
    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.show()

def main(n=10, lr=0.05, lr_dropped=0.005, repeat=20,
         deploy=None, model=None, phase=caffe.TRAIN,
         path="cifar10_train_lmdb", recompute=False):
    net, his, dataset = init_all(deploy, model, phase, path, recompute)
    
    temp_net = "temp_net.caffemodel"
    
    # before dropping
    print "Before dropping"
    step_n_net(net, his, dataset, n=n, lr=lr)
    sgn0 = get_blobs_sgn(net)
    trans_before = np.zeros((repeat, len(sgn0)))
    for r in xrange(repeat):
        print step_n_net(net, his, dataset, n=n, lr=lr)
        sgn0 = get_blobs_sgn(net)
        net.forward()
        sgn1 = get_blobs_sgn(net)
        trans_before[r,:] = np.array([get_trans(s0,s1) for s0,s1 in zip(sgn0, sgn1)])
#    print trans_before
    
    # save temp net
#    net.save(temp_net)
#    old_his = copy.deepcopy(his)
    
    # after dropping
    print "After dropping"
    trans_after = np.zeros_like(trans_before)
    for r in xrange(repeat):
#        net.copy_from(temp_net)
        print step_n_net(net, his, dataset, n=n, lr=lr_dropped)
        sgn0 = get_blobs_sgn(net)
        net.forward()
        sgn1 = get_blobs_sgn(net)
        trans_after[r,:] = np.array([get_trans(s0,s1) for s0,s1 in zip(sgn0, sgn1)])    
#    print trans_after
    return trans_before, trans_after
   
if __name__ == "__main__":
    start_time = time.time()
    main()
    print "Time elapsed %.3f" % (time.time() - start_time)