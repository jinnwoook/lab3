import tqdm
import torch

from torchmetrics.aggregation import MeanMetric


def train_one_epoch_supervised(
        model, loader, metric_fn, loss_fn, device, optimizer, scheduler):
    # set model to train mode    
    model.train()

    # average meters to trace loss and accuracy
    loss_epoch = MeanMetric()
    accuracy_epoch = MeanMetric()    
    
    # train loop
    for inputs, targets in tqdm.tqdm(loader):
        # move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # forward
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        accuracy = metric_fn(outputs, targets)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # update statistics
        loss_epoch.update(loss.to('cpu'))
        accuracy_epoch.update(accuracy.to('cpu'))
    
    summary = {
        'loss': loss_epoch.compute(),
        'accuracy': accuracy_epoch.compute(),
    }

    return summary


def train_one_epoch_semi_supervised(
        model, loader, unlabeled_loader, metric_fn, loss_fn, loss_unlabeled_fn, device, optimizer, scheduler):
    # set model to train mode    
    model.train()

    # average meters to trace loss and accuracy
    loss_epoch = MeanMetric()
    accuracy_epoch = MeanMetric()    
    
    # train loop
    count = 0
    for (inputs, targets), (unlabeled_inputs, unlabeled_targets) in tqdm.tqdm(zip(loader, unlabeled_loader)):
        # move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        unlabeled_inputs = unlabeled_inputs.to(device)
        unlabeled_targets = unlabeled_targets.to(device)

            
        # forward
        outputs = model(torch.cat([inputs, unlabeled_inputs], dim=0))
        outputs, unlabeled_outputs = torch.split(outputs, [inputs.shape[0], unlabeled_inputs.shape[0]])
        
        loss_labeled = loss_fn(outputs, targets)

        # get unlabeled target
        with torch.no_grad():
            unlabeled_targets = model(unlabeled_targets)

        loss_unlabeled = loss_unlabeled_fn(unlabeled_outputs, unlabeled_targets)

        loss = loss_labeled + loss_unlabeled
        accuracy = metric_fn(outputs, targets)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # update statistics
        loss_epoch.update(loss.to('cpu'))
        accuracy_epoch.update(accuracy.to('cpu'))
    
    summary = {
        'loss': loss_epoch.compute(),
        'accuracy': accuracy_epoch.compute(),
    }

    return summary


def eval_one_epoch(model, loader, metric_fn, loss_fn, device):
    # set model to evaluatinon mode    
    model.eval()
    
    # average meters to trace loss and accuracy
    loss_epoch = MeanMetric()
    accuracy_epoch = MeanMetric()    
    
    # train loop
    for inputs, targets in tqdm.tqdm(loader):
        # move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # forward
        with torch.no_grad():
            outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        accuracy = metric_fn(outputs, targets)
        
        # update statistics
        loss_epoch.update(loss.to('cpu'))
        accuracy_epoch.update(accuracy.to('cpu'))
    
    summary = {
        'loss': loss_epoch.compute(),
        'accuracy': accuracy_epoch.compute(),
    }

    return summary