import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SN

class SPADE(nn.Module):
    # seg_channel : # channel of segmentation map
    # main_channel : # channel of main input and output stream channel 
    def __init__(self, seg_channel, main_channel):
        super(SPADE, self).__init__()
        self.seg_channel = seg_channel
        self.main_channel = main_channel
        self.n_hidden = 128 

        self.batch = nn.BatchNorm2d(self.main_channel)

        self.share_cov = nn.Sequential(
            nn.Conv2d(in_channels=self.seg_channel, out_channels=self.n_hidden, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.gamma = nn.Conv2d(in_channels=self.n_hidden, out_channels=self.main_channel, kernel_size=3, stride=1, padding=1)
        self.beta = nn.Conv2d(in_channels=self.n_hidden, out_channels=self.main_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x, seg, seg_resize):
        # x = 'sync_batch_norm'(x) # input channel 
        x = self.batch(x)

        seg = F.interpolate(input=seg, size=seg_resize, mode='nearest')
        seg_share = self.share_cov(seg)
        seg_gamma = self.gamma(seg_share)
        seg_beta = self.beta(seg_share)
        
        # x = x * seg_gamma + seg_beta 
        x = x * (1 + seg_gamma) + seg_beta

        return x


class SPADE_Res(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SPADE_Res, self).__init__()
        segmen_channel = 3 
        self.do_resblk = (in_channel != out_channel) # learn re-sblock when input channel differ from output channel 
        middle_channel = min(in_channel, out_channel)

        self.SPADE_1 = SPADE(seg_channel=segmen_channel, main_channel=in_channel)
        self.SPADE_1_back = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            SN(nn.Conv2d(in_channels=in_channel, out_channels=middle_channel, kernel_size=3, padding=1))
        )
        
        self.SPADE_2 = SPADE(seg_channel=segmen_channel, main_channel=middle_channel)
        self.SPADE_2_back = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            SN(nn.Conv2d(in_channels=middle_channel, out_channels=out_channel, kernel_size=3, padding=1))
        )
        
        if self.do_resblk : 
            self.SPADE_Res = SPADE(seg_channel=segmen_channel, main_channel=in_channel)
            self.SPADE_Res_back = nn.Sequential(
                # nn.LeakyReLU(negative_slope=0.2, inplace=False),
                SN(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=False))
            )


    def forward(self, x, seg, seg_resize):
        main_x = self.SPADE_1(x, seg, seg_resize)
        main_x = self.SPADE_1_back(main_x)
        main_x = self.SPADE_2(main_x, seg, seg_resize)
        main_x = self.SPADE_2_back(main_x)

        if self.do_resblk : 
            res_x = self.SPADE_Res(x, seg, seg_resize)
            res_x = self.SPADE_Res_back(res_x)
        
        else : 
            res_x = x 

        out = main_x + res_x

        return out 

class image_encoder(nn.Module):
    def __init__(self):
        super(image_encoder, self).__init__()

        nf = 64 

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=nf*1, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=nf*1),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),

            nn.Conv2d(in_channels=nf*1, out_channels=nf*2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=nf*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),

            nn.Conv2d(in_channels=nf*2, out_channels=nf*4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=nf*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),

            nn.Conv2d(in_channels=nf*4, out_channels=nf*8, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=nf*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),

            nn.Conv2d(in_channels=nf*8, out_channels=nf*8, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=nf*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),

            nn.Conv2d(in_channels=nf*8, out_channels=nf*8, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=nf*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )

        self.mu_linear = nn.Sequential(
            nn.Linear(in_features=8192, out_features=256),
        )

        self.logvar_linear = nn.Sequential(
            nn.Linear(in_features=8192, out_features=256),
        )

    def reparameterize(self, mu, logvar): 
        std = torch.exp(0.5*logvar)
        normal = torch.randn_like(std)
        
        z = mu + normal.mul(std) 

        return z

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 8192)
        mu = self.mu_linear(x)
        logvar = self.logvar_linear(x)
        
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

        
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()

        self.latent_linear = nn.Linear(in_features=256, out_features=16384)
        
        self.SPADE_Res_4_4 = SPADE_Res(in_channel=1024, out_channel=1024)
        self.SPADE_Res_8_8 = SPADE_Res(in_channel=1024, out_channel=1024)
        self.SPADE_Res_16_16 = SPADE_Res(in_channel=1024, out_channel=1024)
        self.SPADE_Res_32_32 = SPADE_Res(in_channel=1024, out_channel=512)
        self.SPADE_Res_64_64 = SPADE_Res(in_channel=512, out_channel=256)
        self.SPADE_Res_128_128 = SPADE_Res(in_channel=256, out_channel=128)
        self.SPADE_Res_256_256 = SPADE_Res(in_channel=128, out_channel=64)

        self.final_lrelu = nn.LeakyReLU(0.2)
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=2, padding=1)
        self.final_acti = nn.Tanh()

    def forward(self, z, seg):
        z = self.latent_linear(z)
        z = z.view(-1, 1024, 4, 4)

        out = self.SPADE_Res_4_4(z, seg, seg_resize=4)
        out = F.interpolate(out, size=out.shape[2]*2, mode='nearest')
        out = self.SPADE_Res_8_8(out, seg, seg_resize=8)
        out = F.interpolate(out, size=out.shape[2]*2, mode='nearest')
        out = self.SPADE_Res_16_16(out, seg, seg_resize=16)
        out = F.interpolate(out, size=out.shape[2]*2, mode='nearest')
        out = self.SPADE_Res_32_32(out, seg, seg_resize=32)
        out = F.interpolate(out, size=out.shape[2]*2, mode='nearest')
        out = self.SPADE_Res_64_64(out, seg, seg_resize=64)
        out = F.interpolate(out, size=out.shape[2]*2, mode='nearest')
        out = self.SPADE_Res_128_128(out, seg, seg_resize=128)
        out = F.interpolate(out, size=out.shape[2]*2, mode='nearest')
        out = self.SPADE_Res_256_256(out, seg, seg_resize=256)
        out = F.interpolate(out, size=out.shape[2]*2, mode='nearest')

        out = self.final_lrelu(out)
        out = self.final_conv(out) # 512 -> 256 
        out = self.final_acti(out)
    
        return out

        
class discriminator(nn.Module):
    def __init__(self, down_scale):
        super(discriminator, self).__init__()

        nf = 64 

        self.down_scale = down_scale

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=nf*1, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=nf*1, out_channels=nf*2, kernel_size=4, stride=2, padding=2),
            nn.InstanceNorm2d(num_features=nf*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=nf*2, out_channels=nf*4, kernel_size=4, stride=2, padding=2),
            nn.InstanceNorm2d(num_features=nf*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=nf*4, out_channels=nf*8, kernel_size=4, stride=1, padding=2),
            nn.InstanceNorm2d(num_features=nf*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )
        self.out = nn.Sequential(
            # nn.InstanceNorm2d(num_features=nf*8),
            # nn.LeakyReLU(negative_slope=0.2, inplace=False),

            nn.Conv2d(in_channels=nf*8, out_channels=1, kernel_size=4, stride=1, padding=2)
        )

    def split2realNfake(self, mixed_output) :
        half_size = mixed_output.size(0)//2
        (real, fake) = torch.split(mixed_output, half_size, dim=0)

        return real, fake 

    def forward(self, real, fake, seg):
        concat_real = torch.cat((real, seg), dim=1)
        concat_fake = torch.cat((fake, seg), dim=1)

        concat = torch.cat((concat_real, concat_fake), dim=0)
        
        size = int(concat.shape[2]/self.down_scale)
        scaled = F.interpolate(input=concat, size=size, mode='nearest')

        conv1 = self.conv1(scaled)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        out = self.out(conv4)
        
        results = [conv1, conv2, conv3, conv4, out]

        list_real = [] 
        list_fake = [] 

        for result in results: 
            real, fake = self.split2realNfake(result)
            list_real.append(real)
            list_fake.append(fake)

        return list_real, list_fake


def _init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear): 
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None : 
            torch.nn.init.zeros_(m.bias)