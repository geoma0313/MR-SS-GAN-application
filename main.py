import torch,time, random, os, glob
import numpy as np
import pdb
import pandas as pd
import ml_collections
import argparse
from train_util import *
from early_stopping import EarlyStopping
from model_mrssgan import Discriminator, Generator, weights_init
from train_mrssgan import MR_SS_GAN, save_data_proc_gan 

def loaddataAndtrain(args, synth_flag, suffix, mrsssgan=[], G=[], D=[],  m1=[], m2=[], proba=1, plt_layer=0):#proba=绘制概率还是类别, plt_layer=绘制层位图
    storePath='exp_synth%s'%synth_flag
    if not os.path.exists(storePath):
        os.makedirs(storePath)
    
    data_loaders, data_set = get_dataset_synth ( args )            
    mrsssgan = mrsssgan(args.lr, args.train_batch_size, args.latent_dim, args.nClusters, G, D, data_loaders, args.output_dir,args.pati)
    mrsssgan.train(args.eph_sum, synth_flag, suffix)
        
               
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nClusters',type=int,default=2)
    parser.add_argument('--synth',type=int,default=1)
    parser.add_argument("--train_batch_size", default=10, type=int,  help="Total batch size for training.")
    parser.add_argument('--img_f',type=int,default=45)
    parser.add_argument('--img_t',type=int,default=5)
    parser.add_argument('--latent_dim',type=int,default=50)
    parser.add_argument("--output_dir", default="exp_outmodel", type=str, help="The output directory where checkpoints will be written.")
    parser.add_argument("--lr", default=1e-4, type=float, help="The initial learning rate for SGD.")
    parser.add_argument('--pati',type=int,default=5)
    parser.add_argument("--eph_sum", default=50, type=int)   
    args = parser.parse_args()
    
    SEED=0
    setup_seed(SEED)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
         
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)        
    
    img_f,img_t,batch_size = args.img_f,args.img_t,args.train_batch_size
    suffix='synth%s_f%s-t%s_lr%s'%(args.synth, img_f, img_t, args.lr)      
    
    G = Generator(args.latent_dim).apply(weights_init)
    D = Discriminator(args.nClusters).apply(weights_init)
    
    synth_flag=args.synth
    loaddataAndtrain(args, synth_flag, suffix, mrsssgan=MR_SS_GAN, G=G, D=D)
    
    
if __name__ == "__main__":
    main()