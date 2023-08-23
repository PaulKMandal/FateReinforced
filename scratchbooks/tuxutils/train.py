from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms
import torchvision

def train(args):
    from os import path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Planner().to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    """
    Your code here, modify your HW4 code
    Hint: Use the log function below to debug and visualize your model
    """

    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=0.9, weight_decay=1e-5)
    loss = torch.nn.MSELoss()

    import inspect
    #transform = eval(args.transform,
                     #{k: v for k, v in inspect.getmembers(torchvision.transforms) if inspect.isclass(v)})
    transform = dense_transforms.Compose([dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1), dense_transforms.RandomHorizontalFlip(), dense_transforms.ToTensor()])
    train_data = load_data('drive_data', transform=transform, num_workers=4)
    #valid_data = load_data('data/valid', num_workers=4)

    global_step = 0
    num_epoch = 500
    
    for epoch in range(num_epoch):
        model.train()
        #confusion = ConfusionMatrix(len(LABEL_NAMES))
        global_loss = 0
        for img, label in train_data:
            if train_logger is not None:
                train_logger.add_images('augmented_image', img[:4])
            img, label = img.to(device), label.to(device)

            logit = model(img)
            loss_val = loss(logit, label)
            #confusion.add(logit.argmax(1), label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
                
            global_loss += loss_val/len(train_data)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
            
        save_model(model)
        print("Saved at epoch: " + str(epoch))
        print("loss" + str(global_loss))

        """
        if train_logger:
            train_logger.add_scalar('accuracy', confusion.global_accuracy, global_step)
            import matplotlib.pyplot as plt
            f, ax = plt.subplots()
            ax.imshow(confusion.per_class, interpolation='nearest', cmap=plt.cm.Blues)
            for i in range(confusion.per_class.size(0)):
                for j in range(confusion.per_class.size(1)):
                    ax.text(j, i, format(confusion.per_class[i, j], '.2f'),
                            ha="center", va="center", color="black")
            train_logger.add_figure('confusion', f, global_step)
        """
        """
        model.eval()
        val_confusion = ConfusionMatrix(len(LABEL_NAMES))
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            val_confusion.add(model(img).argmax(1), label)

        if valid_logger:
            valid_logger.add_scalar('accuracy', val_confusion.global_accuracy, global_step)
            import matplotlib.pyplot as plt
            f, ax = plt.subplots()
            ax.imshow(val_confusion.per_class, interpolation='nearest', cmap=plt.cm.Blues)
            for i in range(val_confusion.per_class.size(0)):
                for j in range(val_confusion.per_class.size(1)):
                    ax.text(j, i, format(val_confusion.per_class[i, j], '.2f'),
                            ha="center", va="center", color="black")
            valid_logger.add_figure('confusion', f, global_step)

        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, confusion.global_accuracy,
                                                                    val_confusion.global_accuracy))
        """

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor()])')


    args = parser.parse_args()
    train(args)
