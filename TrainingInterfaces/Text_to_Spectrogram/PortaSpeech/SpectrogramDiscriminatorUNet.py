import torch
import torch.nn as nn


def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class SpectrogramDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.D = DiscriminatorNet()
        self.D.apply(weights_init_D)

    def _generator_feedback(self, data_generated, data_real):
        for p in self.D.parameters():
            p.requires_grad = False  # freeze critic

        score_fake, fmap_fake = self.D(data_generated)
        _, fmap_real = self.D(data_real)

        feature_matching_loss = 0.0
        for feat_fake, feat_real in zip(fmap_fake, fmap_real):
            feature_matching_loss += nn.functional.l1_loss(feat_fake, feat_real.detach())

        discr_loss = nn.functional.mse_loss(input=score_fake, target=torch.ones(score_fake.shape, device=score_fake.device), reduction="mean")

        return feature_matching_loss + discr_loss

    def _discriminator_feature_matching(self, data_generated, data_real):
        for p in self.D.parameters():
            p.requires_grad = True  # unfreeze critic
        self.D.train()

        score_fake, _ = self.D(data_generated)
        score_real, _ = self.D(data_real)

        discr_loss = 0.0
        discr_loss = discr_loss + nn.functional.mse_loss(input=score_fake, target=torch.zeros(score_fake.shape, device=score_fake.device), reduction="mean")
        discr_loss = discr_loss + nn.functional.mse_loss(input=score_real, target=torch.ones(score_real.shape, device=score_real.device), reduction="mean")

        return discr_loss

    def calc_discriminator_loss(self, data_generated, data_real):
        return self._discriminator_feature_matching(data_generated.detach(), data_real)

    def calc_generator_feedback(self, data_generated, data_real):
        return self._generator_feedback(data_generated, data_real)


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=2)),
            nn.utils.weight_norm(nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=2)),
            nn.utils.weight_norm(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2)),
            nn.utils.weight_norm(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2)),
            nn.utils.weight_norm(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2)),
            nn.utils.weight_norm(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2)),
        ])

        self.up_out = nn.utils.weight_norm(nn.Conv2d(256, 1, kernel_size=3, stride=1))

        self.down = nn.ModuleList([
            nn.utils.weight_norm(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=2)),
            nn.utils.weight_norm(nn.ConvTranspose2d(256, 64, kernel_size=3, stride=1, padding=2)),
            nn.utils.weight_norm(nn.ConvTranspose2d(128, 32, kernel_size=3, stride=1, padding=2)),
            nn.utils.weight_norm(nn.ConvTranspose2d(64, 16, kernel_size=3, stride=1, padding=2)),
            nn.utils.weight_norm(nn.ConvTranspose2d(32, 8, kernel_size=3, stride=1, padding=2)),
            nn.utils.weight_norm(nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=2)),
        ])
        
        self.down_out = nn.utils.weight_norm(nn.Conv2d(1, 1, kernel_size=3, stride=1))

        self.fc = nn.Linear(8000, 1)  # this needs to be changed everytime the window length is changes. It would be nice if this could be done dynamically.

    def forward(self, y):
        upsample_outputs = list()
        feature_maps = list()
        for d_up in self.up:
            y = d_up(y)
            y = nn.functional.leaky_relu(y, 0.2)
            upsample_outputs.append(y) 

        up_f_map = self.up_out(y)
        feature_maps.append(up_f_map)

        counter = 0
        uo_len = len(upsample_outputs)
        for d_down in self.down:
            print(d_down)
            if counter == 0:
                y = d_down(y)
                y = nn.functional.leaky_relu(y, 0.2)
            if counter >= 1:
                _ = torch.cat((y, upsample_outputs[uo_len - counter -1]), dim=1)
                y = d_down(torch.cat((y, upsample_outputs[uo_len - counter - 1]), dim=1))
                y = nn.functional.leaky_relu(y, 0.2)
            counter+=1
        
        down_f_map = self.down_out(y)
        feature_maps.append(down_f_map)
        
        y = torch.flatten(y, 1, -1)
        y = self.fc(y)

        return y, feature_maps


if __name__ == '__main__':
    d = SpectrogramDiscriminator()
    fake = torch.randn([2, 100, 80])  # [Batch, Sequence Length, Spectrogram Buckets]
    real = torch.randn([2, 100, 80])  # [Batch, Sequence Length, Spectrogram Buckets]

    critic_loss = d.calc_discriminator_loss((fake.unsqueeze(1)), real.unsqueeze(1))
    generator_loss = d.calc_generator_feedback(fake.unsqueeze(1), real.unsqueeze(1))
    print(critic_loss)
    print(generator_loss)