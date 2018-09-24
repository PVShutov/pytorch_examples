



def exp_lr_scheduler(optimizer, iter, lr=0.001, lr_decay_steps=500, factor=0.95):
    lr = lr * (factor**(iter // lr_decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer