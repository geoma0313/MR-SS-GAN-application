import numpy as np
import pandas as pd
import glob, os, math,random,segyio,torch,torchvision
from torch.utils.data import DataLoader,TensorDataset


SEED=0 
g = torch.Generator()
g.manual_seed(SEED)


def get_label(Xnum, ncls, tile,dupn=0,Width=0,sltID=0,welDry=0):   
    synth_flag=int(tile[3])
    Y=np.zeros((Xnum),dtype=float)
    dlt=int(Xnum/ncls)
    for ll in range(ncls):
        Y[dlt*ll:dlt*(ll+1)] = ll
        
    return Y
    
def get_loaderandset(X,Y,batch_size,shuffle=True):
    dataloader=DataLoader(TensorDataset(X,Y),batch_size=batch_size,shuffle=shuffle,num_workers=1,drop_last=True,worker_init_fn=worker_init_fn,generator=g)
    
    dataset=dict()
    dataset['X']=X
    dataset['Y']=Y
    return dataloader,dataset 
        
def create_infinite_dataloader(dataloader):
    data_iter = iter(dataloader)
    while True:
        try:
            yield next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
    
    
def get_dataset_synth(args):
    synth_flag=1
    Fs=250    
    img_f, img_t = args.img_f, args.img_t
    batch_size, ncls=args.train_batch_size, args.nClusters
    
    current_path = os.getcwd()
    tf_file='%s/mar2_FSST_f%s-t%s'% (current_path, img_f, img_t)
    X_lb = np.load(tf_file+'_train_lb.npy')
    maxvalue = np.max( np.max(abs(X_lb), axis=1), axis=0)   
    X_lb=torch.Tensor(X_lb/maxvalue)
    X_lb=torch.unsqueeze(X_lb, 1)
    Y_lb=get_label(X_lb.shape[0], ncls, 'syn1_train_lb')  
    train_lb_DL,train_lb_set = get_loaderandset(X_lb, torch.Tensor(Y_lb).long(), batch_size, shuffle=True)
    train_lb_DL = create_infinite_dataloader(train_lb_DL)
    
    X_ul = np.load(tf_file+'_train_ul.npy')
    X_ul=torch.Tensor(X_ul/maxvalue)
    X_ul=torch.unsqueeze(X_ul, 1)
    Y_ul=get_label(X_ul.shape[0], ncls, 'syn1_train_ul')  
    train_ul_DL,train_ul_set = get_loaderandset(X_ul, torch.Tensor(Y_ul).long(), batch_size, shuffle=True)
    
    X_ts = np.load(tf_file+'_test.npy')
    X_ts=torch.Tensor(X_ts/maxvalue)
    X_ts=torch.unsqueeze(X_ts, 1)
    Y_ts=get_label(X_ts.shape[0], ncls, 'syn1_test')  
    test_DL,test_set = get_loaderandset(X_ts, torch.Tensor(Y_ts).long(), batch_size, shuffle=True)
    
    X_vd = np.load(tf_file+'_valid.npy')
    X_vd=torch.Tensor(X_vd/maxvalue)
    X_vd=torch.unsqueeze(X_vd, 1)
    Y_vd=get_label(X_vd.shape[0], ncls, 'syn1_valid')  
    valid_DL,valid_set = get_loaderandset(X_vd, torch.Tensor(Y_vd).long(), batch_size, shuffle=True)
    # print(X_lb.shape,X_ul.shape,X_ts.shape,X_vd.shape,)
    
    return [train_ul_DL, train_lb_DL, valid_DL, test_DL], [train_ul_set, train_lb_set, valid_set, test_set]
    
def setup_seed(seed):
    np.random.seed(seed) 
    random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)  
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 
    torch.manual_seed(seed)
    
    torch.use_deterministic_algorithms(True) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.enabled = False  
    torch.backends.cudnn.benchmark = False  
                
def worker_init_fn(worker_id):
    random.seed(SEED + worker_id)
