import torch


class FeatureMatchingLoss:
    def __call__(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss * 2


class DiscriminatorLoss:
    def __call__(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg ** 2)
            loss += r_loss + g_loss

        return loss


class GeneratorLoss:
    def __call__(self, disc_outputs):
        loss = 0
        for dg in disc_outputs:
            l = torch.mean((1 - dg) ** 2)
            loss += l

        return loss

