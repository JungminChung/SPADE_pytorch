import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import models

class KL_loss(nn.Module):
    def __init__(self, device):
        super(KL_loss, self).__init__()
        self.device = device 

    def forward(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return(kl_loss)

class GAN_D_loss(nn.Module):
    def __init__(self, device, GAN_D_loss_type):
        super(GAN_D_loss, self).__init__()
        self.device = device
        self.GAN_D_loss_type = GAN_D_loss_type

        assert self.GAN_D_loss_type in ["lsgan", "hinge", "origin"], print("wrong gan D loss type")
    
    def forward(self, list_D_real_img, list_D_fake_img):
        d_real_losses = 0 
        d_fake_losses = 0 
        d_losses = 0 

        for i, (real, fake) in enumerate(zip(list_D_real_img, list_D_fake_img)):
            ones = torch.ones_like(real).to(self.device)
            zeros = torch.zeros_like(fake).to(self.device)

            if self.GAN_D_loss_type == "lsgan": 
                d_real_loss = torch.mean(nn.MSELoss()(real, ones))
                d_fake_loss = torch.mean(nn.MSELoss()(fake, zeros))
            elif self.GAN_D_loss_type == "hinge":
                d_real_loss = -torch.mean(torch.min( real-1, zeros))
                d_fake_loss = -torch.mean(torch.min(-fake-1, zeros))
            elif self.GAN_D_loss_type == "origin":
                d_real_loss = torch.mean(nn.BCEWithLogitsLoss()(real, ones))
                d_fake_loss = torch.mean(nn.BCEWithLogitsLoss()(fake, zeros)) 
                
            d_loss = d_real_loss + d_fake_loss

            d_real_losses += d_real_loss
            d_fake_losses += d_fake_loss
            d_losses += d_loss

        return d_real_losses/(i+1), d_fake_losses/(i+1), d_losses/(i+1) 

class GAN_G_loss(nn.Module):
    def __init__(self, device, GAN_G_loss_type):
        super(GAN_G_loss, self).__init__()
        self.device = device 
        self.GAN_G_loss_type = GAN_G_loss_type

        assert self.GAN_G_loss_type in ["lsgan", "hinge", "origin"], print("wrong gan G loss type")

    def forward(self, list_D_fake_img):
        g_losses = 0 

        for i, fake in enumerate(list_D_fake_img): 
            ones = torch.ones_like(fake).to(self.device)

            if self.GAN_G_loss_type == "lsgan": 
                g_fake_loss = torch.mean(nn.MSELoss()(fake, ones))
            elif self.GAN_G_loss_type == "hinge":
                g_fake_loss = -torch.mean(fake)
            elif self.GAN_G_loss_type == "origin":
                g_fake_loss = torch.mean(nn.BCEWithLogitsLoss()(fake, ones)) 

            g_losses += g_fake_loss

        return g_losses/(i+1)

class FM_loss(nn.Module):
    def __init__(self, device):
        super(FM_loss, self).__init__()
        self.device = device

    def forward(self, listed_FMs_D_real_img, listed_FMs_D_fake_img):
        fm_losses = 0
        for i, (FMs_D_real_img, FMs_D_fake_img) in enumerate(zip(listed_FMs_D_real_img, listed_FMs_D_fake_img)):
            single_list_output = 0 
            for j, (real, fake) in enumerate(zip(FMs_D_real_img, FMs_D_fake_img)):
                loss = torch.mean(torch.abs(real-fake))
                single_list_output += loss 
            fm_losses += single_list_output/(j+1) 
        
        return fm_losses/(i+1)

class VGG_loss(nn.Module):
    def __init__(self, device):
        super(VGG_loss, self).__init__()
        self.device = device

        vgg_pretrained_feaures = models.vgg19(pretrained=True).features.to(self.device)
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_feaures[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_feaures[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_feaures[x])
        for x in range(12, 21): 
            self.slice4.add_module(str(x), vgg_pretrained_feaures[x])
        for x in range(21, 30): 
            self.slice5.add_module(str(x), vgg_pretrained_feaures[x])

        for param in self.parameters():
            param.requires_grad= False

    def forward(self, real_image, fake_image):
        loss = 0 

        real_h = self.slice1(real_image)
        fake_h = self.slice1(fake_image)
        loss += torch.mean(torch.abs(real_h-fake_h)) * 1/32

        real_h = self.slice2(real_h)
        fake_h = self.slice2(fake_h)
        loss += torch.mean(torch.abs(real_h-fake_h)) * 1/16

        real_h = self.slice3(real_h)
        fake_h = self.slice3(fake_h)
        loss += torch.mean(torch.abs(real_h-fake_h)) * 1/8

        real_h = self.slice4(real_h)
        fake_h = self.slice4(fake_h)
        loss += torch.mean(torch.abs(real_h-fake_h)) * 1/4

        real_h = self.slice5(real_h)
        fake_h = self.slice5(fake_h)
        loss += torch.mean(torch.abs(real_h-fake_h)) * 1

        return loss

    def denorm(self, x):
        return ((x+1)/2) * 255.0