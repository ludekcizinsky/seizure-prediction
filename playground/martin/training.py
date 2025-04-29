import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict

def train_epoch(model, device, loader_tr, optimizer, criterion, epoch, metrics, verbose=True, edge_index=None):
    model.train()
    # iterate over batches in training set
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    for x, y in tqdm(loader_tr, desc=f"Epoch {epoch +1}"):
        
        x = x.to(torch.float32).to(device)
        y = y.to(torch.float32).to(device)
        
        optimizer.zero_grad()
        if edge_index is not None:
            outputs = model(x, edge_index)
        else:
            outputs = model(x)

        with torch.no_grad():
            preds = torch.round(torch.sigmoid(outputs))
            epoch_acc += accuracy_score(preds.cpu(), y.cpu())
            epoch_f1 += f1_score(preds.cpu(), y.cpu())

        loss = criterion(outputs.view(-1),y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(loader_tr)
    epoch_acc /= len(loader_tr)
    epoch_f1 /= len(loader_tr)

    metrics["train"]["loss"].append(epoch_loss)
    metrics["train"]["acc"].append(epoch_acc)
    metrics["train"]["f1"].append(epoch_f1)
    if verbose:
        print('Train Loss: {:.4f}, '.format(epoch_loss),'Train Accuracy: {:.4f}, '.format(epoch_acc),'Train F1-Score: {:.4f}, '.format(epoch_f1))

def eval_epoch(model, device, loader_val, criterion, metrics, verbose=True, edge_index=None):
    model.eval()
    eval_loss = 0
    eval_acc = 0
    eval_f1 = 0
    for x, y in loader_val:
        with torch.no_grad():
            
            x = x.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device)

            if edge_index is not None:
                outputs = model(x, edge_index)
            else:
                outputs = model(x)

            loss = criterion(outputs.view(-1), y)
            preds = torch.round(torch.sigmoid(outputs))
        
            eval_loss += loss.item()
            eval_acc += accuracy_score(preds.cpu(), y.cpu())
            eval_f1 += f1_score(preds.cpu(), y.cpu())
            
    eval_loss /= len(loader_val)
    eval_acc /= len(loader_val)
    eval_f1 /= len(loader_val)

    metrics["eval"]["loss"].append(eval_loss)
    metrics["eval"]["acc"].append(eval_acc)
    metrics["eval"]["f1"].append(eval_f1)

    if verbose:
        print('Val Loss: {:.4f}, '.format(eval_loss),'Val Accuracy: {:.4f}, '.format(eval_acc),'Val F1-Score: {:.4f}, '.format(eval_f1))

def train(model, num_epochs, device, loader_tr, loader_val, optimizer, criterion, model_folder, verbose=True, edge_index=None):
    metrics = dict()
    metrics["train"] = defaultdict(list)
    metrics["eval"] = defaultdict(list)

    best_f1 = 0.0

    # iterate over epochs
    for epoch in range(num_epochs):

        train_epoch(model, device, loader_tr, optimizer, criterion, epoch, metrics, verbose, edge_index)
        eval_epoch(model, device, loader_val, criterion, metrics, verbose, edge_index)

        current_f1 = metrics["eval"]["f1"][-1]
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), f"{model_folder}/best_model.pt")
            print(f"✅ Best model saved with F1: {best_f1:.4f} as {model_folder}/best_model.pt")
        
    return metrics