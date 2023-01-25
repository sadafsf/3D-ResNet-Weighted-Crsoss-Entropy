
import os
import numpy as np
import torch

from utils import adjust_learning_rate, progress_bar

def train_crossentropy_no_proj(train_loader, model, criterion, optimizer, writer, args):

    """
    :param model: torch.nn.Module Model
    :param train_loader: torch.utils.data.DataLoader
    :param test_loader: torch.utils.data.DataLoader
    :param criterion: torch.nn.Module Loss
    :param optimizer: torch.optim
    :param writer: torch.utils.tensorboard.SummaryWriter
    :param args: argparse.Namespace
    :return:
    """

    for epoch in range(args.n_epochs_cross_entropy):  # loop over the dataset multiple times
        print("Epoch [%d/%d]" % (epoch + 1, args.n_epochs_cross_entropy))

        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(args.device), targets.long().squeeze().to(args.device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # _, predicted = outputs.max(1)

            # total_batch = targets.size(0)
            # correct_batch = predicted.eq(targets).sum().item()
            # total += total_batch
            # correct += correct_batch
            preds = torch.max(softmax(outputs), 1)[1]
            y_trues = np.append(y_trues, labels.data.cpu().numpy())
            y_preds = np.append(y_preds, preds.cpu())

            writer.add_scalar(
                "Loss train | Cross Entropy",
                loss.item(),
                epoch * len(train_loader) + batch_idx,
            )
            writer.add_scalar(
                "Accuracy train | Cross Entropy",
                accuracy_score(y_trues, y_preds),
                epoch * len(train_loader) + batch_idx,
            )
            progress_bar(
                batch_idx,
                len(train_loader),
                "Loss: %.3f | Acc: %.3f%% "
                % (
                    train_loss / (batch_idx + 1),
                    accuracy_score(y_trues, y_preds)
                ),
            )
        acc = 100.0 * correct / total
        writer.add_scalar("Accuracy train per epoch | Cross Entropy", acc, epoch)

        if acc > args.best_acc:
            print("Saving..")
            state = {
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "acc": acc,
                "epoch": epoch,
            }
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")
            torch.save(state, "./checkpoint/ckpt_cross_entropy.pth")
            args.best_acc = acc


        adjust_learning_rate(optimizer, epoch, mode='cross_entropy', args=args)
    print("Finished Training")

