import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch, glob, os
from torch.autograd import Variable
from early_stopping import EarlyStopping

class MR_SS_GAN():
    def __init__(self, lr, batch_size, latent_dim, num_classes, generator, discriminator, data_loaders,output_dir,pati):

        self.batch_size = batch_size
        self.batch_size_cuda = torch.tensor(self.batch_size).cuda()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.lr = lr

        self.save_path = output_dir

        self.G = generator.cuda()
        self.D = discriminator.cuda()
        self.train_unl_loader, self.train_lb_loader, self.valid_loader, self.test_loader = data_loaders

        self.ce_criterion = nn.CrossEntropyLoss().cuda()
        self.mse = nn.MSELoss().cuda()

        self.pati=pati
        self.early_stopping = EarlyStopping(self.save_path,patience=self.pati)                  

    def train(self, num_epochs, synth_flag, suffix):
        opt_G = torch.optim.Adam(self.G.parameters(), lr=self.lr)
        opt_D = torch.optim.Adam(self.D.parameters(), lr=self.lr)
        
        train_D_loss_list, train_G_loss_list, val_D_loss_list, val_D_acc_list, test_D_loss_list, test_D_acc_list=[], [], [], [], [], []
        eph_best = 0
        for epoch_idx in range(num_epochs):
            train_G_loss = train_D_loss = 0
            flops, params = 0, 0
            self.G.train()
            self.D.train()
            for unl_train_x, __ in self.train_unl_loader:
                lb_train_x, lb_train_y = next(self.train_lb_loader)
                unl_train_x = unl_train_x.cuda()
                lb_train_x = lb_train_x.cuda()
                lb_train_y = lb_train_y.cuda()
                   
                z, z_perturbed = self.define_noise()

                # Train Discriminator
                opt_D.zero_grad()
                imgs_fake = self.G(z)
                imgs_fake_perturbed = self.G(z_perturbed)

                features_fake, logits_fake = self.D(imgs_fake)
                features_fake_pertubed, __ = self.D(imgs_fake_perturbed)
                __, logits_lb = self.D(lb_train_x)
                __, logits_unl = self.D(unl_train_x)
                
                logits_sum_unl = torch.logsumexp(logits_unl, dim=1)
                logits_sum_fake = torch.logsumexp(logits_fake, dim=1)
                loss_unsupervised = torch.mean(F.softplus(logits_sum_unl)) - torch.mean(logits_sum_unl) + torch.mean(
                    F.softplus(logits_sum_fake))
                
                loss_supervised = torch.mean(self.ce_criterion(logits_lb, lb_train_y))
                loss_manifold_reg = self.mse(features_fake, features_fake_pertubed) \
                                    / self.batch_size_cuda

                loss_D = loss_supervised + .5 * loss_unsupervised + 1e-3 * loss_manifold_reg
                
                loss_D.backward()
                opt_D.step()
                train_D_loss += loss_D

                # Train Generator
                opt_G.zero_grad()
                opt_D.zero_grad()
                imgs_fake = self.G(z)
                features_fake, __ = self.D(imgs_fake)
                features_real, __ = self.D(unl_train_x)
                m1 = torch.mean(features_real, dim=0)
                m2 = torch.mean(features_fake, dim=0)
                loss_G = torch.mean((m1 - m2) ** 2)  # Feature matching
                loss_G.backward()
                opt_G.step()
                train_G_loss += loss_G
       
            # Evaluate
            train_G_loss /= len(self.train_unl_loader)
            train_D_loss /= len(self.train_unl_loader)

            val_D_acc, val_D_loss, all_truth, all_preds,all_probs   = self.eval(suffix) ##默认是验证模式
            test_D_acc, test_D_loss, all_truth, all_preds,all_probs = self.eval(suffix, test=True) ##

            print('Epoch %d disc_loss %.3f gen_loss %.3f val_D_loss %.3f acc %.3f test_D_loss %.3f acc %.3f' % (
                epoch_idx, train_D_loss, train_G_loss, val_D_loss, val_D_acc, test_D_loss,test_D_acc))
            
            train_D_loss_list.append(train_D_loss.item())
            train_G_loss_list.append(train_G_loss.item())
            val_D_loss_list.append(val_D_loss)
            val_D_acc_list. append(val_D_acc)
            test_D_loss_list.append(test_D_loss)
            test_D_acc_list. append(test_D_acc)
            
            save_data_proc_gan(synth_flag, suffix,'semigan_train', train_D_loss_list, train_G_loss_list, val_D_loss_list, val_D_acc_list,test_D_acc_list, test_D_loss_list)
            
            torch.save(self.D.state_dict(), self.save_path + '/disc_{}_eph{}.pth'.format(suffix, epoch_idx))
            torch.save(self.G.state_dict(), self.save_path + '/gen_{}_eph{}.pth'.format(suffix, epoch_idx))
                    
            self.early_stopping(val_D_loss, self.D, suffix, epoch_idx)#           
            if self.early_stopping.early_stop:#
                print("Early stopping")
                break 
                                
    def eval(self, suffix, test=False, epoch_idx=None):
        if test:
            eval_loader = self.test_loader
        else:
            eval_loader = self.valid_loader

        val_loss = corrects = total_samples = 0.0
        predictions, probabilitys, ground_truth = [], [], []
        with torch.no_grad():
            self.D.eval()
            for x, y in eval_loader:
                x, y = x.cuda(), y.cuda()
                __, logits = self.D(x)
                loss = self.ce_criterion(logits, y)
                val_loss += loss.item()
                out=torch.softmax(logits,dim=1)
                proba,pred_y= out.max(dim=1)# 
                predictions.extend(pred_y.cpu().numpy())
                probabilitys.extend(out.cpu().numpy())
                ground_truth.extend(y.cpu().numpy())
                
                corrects += torch.sum(pred_y == y)
                total_samples += len(y)

            all_preds = np.asarray(predictions)
            all_probs = np.asarray(probabilitys)
            all_truth = np.asarray(ground_truth)
            val_loss /= len(self.valid_loader)
            acc = corrects.item() / total_samples
            
        return acc, val_loss, all_truth,all_preds,all_probs
        
    def pred(self, pred_DL, batch_size, probability=0):
        preddata=np.zeros((batch_size*len(pred_DL)),dtype=float)
        n_end=0
        with torch.no_grad():
            self.D.eval()
            for i, (x, y) in enumerate(pred_DL):
                x, y = x.cuda(), y.cuda()
                __, logits = self.D(x)
                
                out=torch.softmax(logits,dim=1)
                proba,pred_y= out.max(dim=1)# 
                
                if probability==0:
                    val=pred_y.detach().cpu().numpy()
                elif probability==1:
                    val=pred_y.detach().cpu().numpy()*proba.detach().cpu().numpy()
                n_end = n_end + len(val)
                preddata[batch_size*i:n_end] = val
        
        return preddata[0:n_end]

    def define_noise(self):
        z = torch.randn(self.batch_size, self.latent_dim).cuda()
        z_perturbed = z + torch.randn(self.batch_size, self.latent_dim).cuda() * 1e-5
        return z, z_perturbed
        
def save_data_proc_gan(synth_flag,suffix, process, train_D_loss_list, train_G_loss_list, val_D_loss_list, val_D_acc_list, test_D_acc_list=[], test_D_loss_list=[]):
    if process=='semigan_train':
        data = pd.DataFrame({'train_D_loss':train_D_loss_list, 'train_G_loss':train_G_loss_list, 'val_D_losss':val_D_loss_list, 'val_D_acc':val_D_acc_list, 'test_D_losss':test_D_loss_list, 'test_D_acc':val_D_acc_list})
    elif test_D_acc_list!=[]:
        data = pd.DataFrame({'test_D_loss':test_D_loss_list, 'test_D_acc':test_D_acc_list})
    data.to_csv('exp_synth%s/%s_%s_data.csv'%(synth_flag,suffix,process),index = None,encoding = 'utf8')    
    