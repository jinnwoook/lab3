import json
import torch

def save_json(data, path):
    with open(path, 'wt') as fp:
        json.dump(data, fp, indent=4)

def load_json(path):
    with open(path, 'rt') as fp:
        return json.load(fp)

def save_model(path, model, optimizer, scheduler, epoch):
    state_dict = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state_dict, path)