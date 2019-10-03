import torch
import time
import os
import utils

import networks
from loss import *
from networks import *
from dataset import *

from tqdm import tqdm 

import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from tensorboardX import SummaryWriter

class GAUGAN(object):
    def __init__(self, args):
        self.device = args.device
        self.img_path = args.img_path
        self.seg_path = args.seg_path 
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.lr_G = args.lr_G
        self.lr_D = args.lr_D
        self.beta_1 = args.beta_1
        self.beta_2 = args.beta_2
        self.total_step = args.total_step
        self.n_critic = args.n_critic 
        self.n_save = args.n_save
        self.ckpt_dir = args.ckpt_dir
        self.lambda_fm = args.lambda_fm
        self.lambda_kl =args.lambda_kl
        self.lambda_vgg =args.lambda_vgg
        self.grid_n_row = args.grid_n_row
        # self.grid_n_col = args.grid_n_col
        self.n_save_image = args.n_save_image
        self.img_dir = args.img_dir
        self.GAN_D_loss_type = args.GAN_D_loss_type
        self.GAN_G_loss_type = args.GAN_G_loss_type
        self.save_Dis = args.save_Dis
        self.start_annealing_epoch = args.start_annealing_epoch
        self.end_annealing_epoch = args.end_annealing_epoch
        self.end_lr = args.end_lr
        self.test_img_path = args.test_img_path
        self.test_seg_path = args.test_seg_path
        self.test_batch_size = args.test_batch_size
        self.use_vgg = args.use_vgg
        self.n_summary = args.n_summary
        self.sum_dir = args.sum_dir
        self.seg_channel = args.seg_channel

    def load_dataset(self):
        self.transform_img = transforms.Compose([
                                transforms.Resize(size=256, interpolation=0),
                                transforms.ToTensor(),
                            ])

        self.dataset = customDataset(origin_path=self.img_path, 
                                    segmen_path=self.seg_path, 
                                    transform=self.transform_img,
                                )

        self.loader = DataLoader(dataset=self.dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=True, 
                                    num_workers=self.num_workers,
                                    drop_last=True,
                                )

        self.test_dataset = customDataset(origin_path=self.test_img_path, 
                                    segmen_path=self.test_seg_path, 
                                    transform=self.transform_img,
                                )

        self.test_loader = DataLoader(dataset=self.test_dataset, 
                                    batch_size=self.test_batch_size
                                )

    def build_model(self): 
        ########## Networks ##########
        # self.enc = nn.DataParallel(image_encoder()).to(self.device)
        # self.gen = nn.DataParallel(generator()).to(self.device)
        # self.disORI = nn.DataParallel(discriminator(down_scale=1)).to(self.device)
        # self.disHAL = nn.DataParallel(discriminator(down_scale=2)).to(self.device)
        # self.disQUA = nn.DataParallel(discriminator(down_scale=4)).to(self.device)
        self.enc = image_encoder().to(self.device)
        self.gen = generator(seg_channel=self.seg_channel).to(self.device)
        self.disORI = discriminator(down_scale=1).to(self.device)
        self.disHAL = discriminator(down_scale=2).to(self.device)
        # self.disQUA = discriminator(down_scale=4).to(self.device)

        ########## Init Networks with Xavier normal ##########
        self.enc.apply(networks._init_weights)
        self.gen.apply(networks._init_weights)
        self.disORI.apply(networks._init_weights)
        self.disHAL.apply(networks._init_weights)
        # self.disQUA.apply(networks._init_weights)

        # ########## Average Weighted Networks and Setting ##########
        # self.enc_ave = image_encoder().to(self.device)
        # self.gen_ave = generator().to(self.device)
        # # self.accumulate(self.enc_ave, self.enc, 0)
        # # self.accumulate(self.gen_ave, self.gen, 0)
        # self.enc_ave.load_state_dict(self.enc.state_dict())
        # self.gen_ave.load_state_dict(self.gen.state_dict())
        # self.enc_ave.train(False)
        # self.gen_ave.train(False)

        ########## Loss ##########
        self.KLloss = KL_loss(self.device)
        self.GAN_D_loss = GAN_D_loss(self.device, GAN_D_loss_type=self.GAN_D_loss_type)
        self.GAN_G_loss = GAN_G_loss(self.device, GAN_G_loss_type=self.GAN_G_loss_type)
        self.FMloss = FM_loss(self.device)
        if self.use_vgg : 
            self.VGGloss = VGG_loss(self.device)

        ########## Optimizer ##########
        self.G_optim = torch.optim.Adam(list(self.gen.parameters()) + 
                                        list(self.enc.parameters()),
                                        lr=self.lr_G, 
                                        betas=(self.beta_1, self.beta_2)
                                        )
        self.G_lambda = lambda epoch : max(self.end_lr, (epoch-self.start_annealing_epoch)*(self.lr_G-self.end_lr)/(self.start_annealing_epoch-self.end_annealing_epoch)+self.lr_G)
        self.G_optim_sch = torch.optim.lr_scheduler.LambdaLR(self.G_optim, lr_lambda=self.G_lambda)

        self.D_optim = torch.optim.Adam(list(self.disORI.parameters())
                                        + list(self.disHAL.parameters())
                                        # + list(self.disQUA.parameters())
                                        ,
                                        lr=self.lr_D, 
                                        betas=(self.beta_1, self.beta_2)
                                        )
        self.D_lambda = lambda epoch : max(self.end_lr, (epoch-self.start_annealing_epoch)*(self.lr_D-self.end_lr)/(self.start_annealing_epoch-self.end_annealing_epoch)+self.lr_D)
        self.D_optim_sch = torch.optim.lr_scheduler.LambdaLR(self.D_optim, lr_lambda=self.D_lambda)

    def train(self):
        self.enc.train() 
        self.gen.train()
        self.disORI.train() 
        self.disHAL.train()
        # self.disQUA.train()

        summary = SummaryWriter(self.sum_dir)

        data_loader = iter(self.loader)

        test_data_loader = iter(self.test_loader)
        test_real_image, test_seg = next(test_data_loader)
        test_real_image, test_seg = test_real_image.to(self.device), test_seg.to(self.device)
    
        pbar = tqdm(range(self.total_step))
        epoch = 0 

        for step in pbar:
            try : 
                real_image, seg = next(data_loader)

            except : 
                data_loader = iter(self.loader)
                real_image, seg = next(data_loader)
                epoch += 1 
                if epoch >= self.start_annealing_epoch : 
                    self.G_optim_sch.step()
                    self.D_optim_sch.step()

            real_image, seg = real_image.to(self.device), seg.to(self.device)

            ##### train Discriminator #####
            self.D_optim.zero_grad()
        
            mu, squ_sigma, z = self.enc(real_image)
            fake_image = self.gen(z, seg)

            list_real_ORI, list_fake_ORI = self.disORI(real_image, fake_image, seg)
            list_real_HAL, list_fake_HAL = self.disHAL(real_image, fake_image, seg)
            # _, D_real_ORI = self.disORI(real_image, seg)
            # _, D_real_HAL = self.disHAL(real_image, seg)
            # _, D_real_QUA = self.disQUA(real_image, seg)

            # _, D_fake_ORI = self.disORI(fake_image, seg)
            # _, D_fake_HAL = self.disHAL(fake_image, seg)
            # _, D_fake_QUA = self.disQUA(fake_image, seg)

            listed_D_real = [list_real_ORI[-1], list_real_HAL[-1]]
            listed_D_fake = [list_fake_ORI[-1], list_fake_HAL[-1]]
            # listed_D_real = [D_real_ORI, D_real_HAL, D_real_QUA]
            # listed_D_fake = [D_fake_ORI, D_fake_HAL, D_fake_QUA]

            GAN_D_real_loss, GAN_D_fake_loss, GAN_D_loss = self.GAN_D_loss(listed_D_real, listed_D_fake)

            GAN_D = GAN_D_loss 
            GAN_D.backward()
            self.D_optim.step()
            
            ##### train Generator #####
            if step % self.n_critic == 0 :
                self.G_optim.zero_grad()

                mu, squ_sigma, z = self.enc(real_image)
                fake_image = self.gen(z, seg)

                list_real_ORI, list_fake_ORI = self.disORI(real_image, fake_image, seg)
                list_real_HAL, list_fake_HAL = self.disHAL(real_image, fake_image, seg)
                # FMs_real_ORI, D_real_ORI = self.disORI(real_image, seg)
                # FMs_real_HAL, D_real_HAL = self.disHAL(real_image, seg)
                # FMs_real_QUA, D_real_QUA = self.disQUA(real_image, seg)

                # FMs_fake_ORI, D_fake_ORI = self.disORI(fake_image, seg)
                # FMs_fake_HAL, D_fake_HAL = self.disHAL(fake_image, seg)
                # FMs_fake_QUA, D_fake_QUA = self.disQUA(fake_image, seg)

                listed_FMs_D_real = [list_real_ORI[:-1], list_real_HAL[:-1]]
                listed_FMs_D_fake = [list_fake_ORI[:-1], list_fake_HAL[:-1]]
                # listed_FMs_D_real = [FMs_real_ORI, FMs_real_HAL, FMs_real_QUA]
                # listed_FMs_D_fake = [FMs_fake_ORI, FMs_fake_HAL, FMs_fake_QUA]

                listed_D_real = [list_real_ORI[-1], list_real_HAL[-1]]
                listed_D_fake = [list_fake_ORI[-1], list_fake_HAL[-1]]
                # listed_D_real = [D_real_ORI, D_real_HAL, D_real_QUA]
                # listed_D_fake = [D_fake_ORI, D_fake_HAL, D_fake_QUA]

                GAN_G_loss = self.GAN_G_loss(listed_D_fake)
                fm_loss = self.FMloss(listed_FMs_D_real, listed_FMs_D_fake)
                # kl_loss = 0
                kl_loss = self.KLloss(mu, squ_sigma)
                if self.use_vgg : 
                    vgg_loss = self.VGGloss(real_image, fake_image)
                else : 
                    vgg_loss = 0
                
                GAN_G = GAN_G_loss + self.lambda_fm * fm_loss + self.lambda_kl * kl_loss 
                if self.use_vgg : 
                    GAN_G = GAN_G_loss + self.lambda_fm * fm_loss + self.lambda_kl * kl_loss + self.lambda_vgg * vgg_loss
                GAN_G.backward()
                self.G_optim.step()
                # self.accumulate(self.enc_ave, self.enc)
                # self.accumulate(self.gen_ave, self.gen)

            if step % self.n_save == 0 : 
                self.save_ckpt(self.ckpt_dir, step, epoch, self.save_Dis)
 
            if step % self.n_save_image == 0 : 
                fake_image_ref, fake_image_latent = self.eval(test_seg, test_real_image)
                self.save_img(self.img_dir, test_seg, test_real_image, fake_image_ref, fake_image_latent, step)
                print()

            if step % self.n_summary == 0: 
                self.writeLogs(summary, step, GAN_D_real_loss, GAN_D_fake_loss, GAN_D, GAN_G, GAN_G_loss, self.lambda_fm * fm_loss, self.lambda_kl * kl_loss, self.lambda_vgg * vgg_loss)

            state_msg = (
                'Epo : {} ; '.format(epoch) + 
                'D_real : {:0.3f} ; D_fake : {:0.3f} ; '.format(GAN_D_real_loss, GAN_D_fake_loss) + 
                'total_D : {:0.3f} ; total_G : {:0.3f} ; '.format(GAN_D, GAN_G) + 
                'G : {:0.3f} ; FM : {:0.3f} ; '.format(GAN_G_loss, self.lambda_fm * fm_loss) + 
                'kl : {:0.3f} ; vgg : {:0.3f} ;'.format(self.lambda_kl * kl_loss, self.lambda_vgg * vgg_loss) 
            )

            pbar.set_description(state_msg)

    def writeLogs(self, summary, step, D_real, D_fake, D, G, GAN_G, fm, kl, vgg):
        summary.add_scalar('D_real', D_real.item(), step)
        summary.add_scalar('D_fake', D_fake.item(), step)
        summary.add_scalar('D', D.item(), step)
        summary.add_scalar('G', G.item(), step)
        summary.add_scalar('GAN_G', GAN_G.item(), step)
        summary.add_scalar('fm', fm.item(), step)
        summary.add_scalar('kl', kl.item(), step)
        summary.add_scalar('vgg', vgg.item(), step)

    def save_ckpt(self, dir, step, epoch, save_Dis):
        model_dict = {} 
        model_dict['enc'] = self.enc.state_dict()
        model_dict['gen'] = self.gen.state_dict()
        if save_Dis : 
            model_dict['disORI'] = self.disORI.state_dict()
            model_dict['disHAL'] = self.disHAL.state_dict()
            model_dict['disQUA'] = self.disQUA.state_dict()
        torch.save(model_dict, os.path.join(dir, f'{str(step+1).zfill(7)}.ckpt'))


    def save_img(self, dir, seg, real_img, fake_img_ref, fake_img_latent, step):
        image_grid = torch.unsqueeze(seg[0], dim=0)
        for i in range(self.grid_n_row): 
            if i == 0 : 
                image_grid = torch.cat((image_grid, torch.unsqueeze(real_img[i], dim=0)), dim=0)
                image_grid = torch.cat((image_grid, torch.unsqueeze(fake_img_ref[i], dim=0)),dim=0)
                image_grid = torch.cat((image_grid, torch.unsqueeze(fake_img_latent[i], dim=0)),dim=0)
            else : 
                image_grid = torch.cat((image_grid, torch.unsqueeze(seg[i], dim=0)), dim=0)
                image_grid = torch.cat((image_grid, torch.unsqueeze(real_img[i], dim=0)), dim=0)
                image_grid = torch.cat((image_grid, torch.unsqueeze(fake_img_ref[i], dim=0)),dim=0)
                image_grid = torch.cat((image_grid, torch.unsqueeze(fake_img_latent[i], dim=0)), dim=0)
            if i == self.test_batch_size-1 : 
                break

        
        image_grid = make_grid(image_grid, nrow=4, padding=2)
        save_image(image_grid, os.path.join(dir, f'{str(step+1).zfill(7)}.jpg'))

    def load(self, dir, ckpt_step, save_Dis):
        model_dict = torch.load(os.path.join(dir, f'{str(ckpt_step).zfill(7)}.ckpt'))
        self.enc.load_state_dict(model_dict['enc'])
        self.gen.load_state_dict(model_dict['gen'])
        if save_Dis:
            self.disORI.load_state_dict(model_dict['disORI'])
            self.disHAL.load_state_dict(model_dict['disHAL'])
            self.disQUA.load_state_dict(model_dict['disQUA'])

    def eval(self, seg, real_image):
        _, _, z = self.enc(real_image)
        fake_image_ref = self.gen(z, seg)

        latent = torch.randn(self.test_batch_size, 256).to(self.device)
        fake_image_latent = self.gen(latent, seg)

        return fake_image_ref, fake_image_latent

    def test(self):
        pass 
