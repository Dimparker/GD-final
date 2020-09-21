from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR

def cos_lr_scheduler(optimizer,t_mult=5):
    return CosineAnnealingWarmRestarts(optimizer, t_mult)

def exp_lr_scheduler(optimizer, step_size=5):
    return StepLR(optimizer, step_size, gamma=0.9)