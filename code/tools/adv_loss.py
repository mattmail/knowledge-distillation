import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self,  in_channels, mid_channels, mid2_channels):
        super(Discriminator, self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(mid_channels, mid2_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(mid2_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size()[0],x.size()[1], -1)
        x = self.classify(x)
        x = x.view(-1)
        return x

#set model for train

disc = Discriminator(240,120,60)
bce = nn.BCELoss()

if torch.cuda.is_available():
    disc = disc.cuda()
    bce = bce.cuda()


disc_params = disc.parameters()
disc_optimizer = optim.Adam(disc_params, lr=0.0002, betas=(0.5, 0.999))
disc_weight = 1

#main train
disc = disc.train()

preds_target = disc(bneck_target)
preds_source = disc(bneck_source)

A_label = torch.full((preds_target.size()), 1)
B_label = torch.full((preds_source.size()), 0)

if torch.cuda.is_available():
    A_label = A_label.cuda()
    B_label = B_label.cuda()

distribution_adverserial_loss = disc_weight * (bce(preds_target, A_label) + bce(preds_source, A_label))
KL_loss += distribution_adverserial_loss
KL_loss.backward()
torch.nn.utils.clip_grad_norm_(main.parameters(), 5) #"main" is because I do not know how you call your main net
main_optimizer.step() #main_optimizer is because I do not know how you call your optimizer

#train discriminator just after main train
disc_optimizer.zero_grad()
disc_target = disc(bneck_target)
disc_source = disc(bneck_source)
disc_loss = bce(disc_target, B_label) + bce(disc_source, A_label)
disc_loss.backward()
torch.nn.utils.clip_grad_norm_(disc_params, 5)
disc_optimizer.step()

#The discriminator attempts to separate between the distributions by classifying samples of target as 0 and the samples of source as 1,
#whereas the bottlenecks tries to fool the discriminator, hence forcing both distributions to match.

#Adapted from {Benaim2019DomainIntersectionDifference, title={Domain Intersection and Domain Difference},
# author={Sagie Benaim and Michael Khaitov and Tomer Galanti and Lior Wolf}, booktitle={ICCV}, year={2019}}