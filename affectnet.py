import argparse
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import datetime
from tqdm import tqdm
from model.main import RARN
from utils.util import *
from utils.dataset import *
from utils.imbalanced import *


now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")
time_s = now.strftime("[%m-%d]-[%H-%M]")
checkpoint_path = './checkpoint/'+ time_s + '/'
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
# data_path = './data/RAF-DB'
# checkpoint_path = ''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/raid/name/lixiao/datasets/affectnet', help='AfectNet dataset path.')
    parser.add_argument('--num_class', type=int, default=8, help='Number of class.')
    parser.add_argument('--checkpoint_path', type=str, default=checkpoint_path + time_str + 'model.pth')
    parser.add_argument('--best_checkpoint_path', type=str, default=checkpoint_path +time_str +'model_best.pth')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', dest='lr')  # 0.1
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=400, type=int, metavar='N', help='print frequency')
    parser.add_argument('--resume', default='./checkpoint/', type=str, metavar='PATH', help='path to checkpoint')
    parser.add_argument('-e', '--evaluate', default=False, action='store_true', help='evaluate model on test set')
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--gpu', type=str, default='4')
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu   # gpu
    print('Training time: ' + now.strftime("%m-%d %H:%M"))
    
    model = RARN()
    model = torch.nn.DataParallel(model).cuda()  # model.to(device)
    checkpoint = torch.load('/raid/name/lixiao/self_ckpt/checkpoint/Pretrained_on_MSCeleb.pth.tar')
    pre_trained_dict = checkpoint['state_dict']
    model.load_state_dict(pre_trained_dict, False)
    model.module.fc_1 = torch.nn.Linear(128, 7).cuda()
    
    # 1/define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_af = RALoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)   # step_size=10
    recorder = RecorderMeter(args.epochs)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            best_acc = best_acc.to()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # 2/Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')

    train_dataset = datasets.ImageFolder(traindir,
                                         transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor()]))
    print('Whole train set size:', train_dataset.__len__())

    test_dataset = datasets.ImageFolder(valdir,
                                        transforms.Compose([transforms.Resize((224, 224)),
                                                            transforms.ToTensor()]))
    
    print('Validation set size:', test_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               sampler=ImbalancedDatasetSampler(train_dataset),
                                               num_workers=args.workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)
    
    # only evaluate
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    best_acc = 0
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        print('Current learning rate: ', current_learning_rate)
        txt_name = checkpoint_path + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')
    
        # train for one epoch
        train_acc, train_los = train(train_loader, model, criterion, criterion_af, optimizer, epoch, args)
    
        # evaluate on validation set
        val_acc, val_los = validate(val_loader, model, criterion, criterion_af, args)
    
        scheduler.step()
    
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        curve_name = time_str + 'cnn.png'
        recorder.plot_curve(os.path.join(checkpoint_path, curve_name))
    
        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
    
        print('Current best accuracy: ', best_acc.item())
        txt_name = checkpoint_path + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc.item()) + '\n')
    
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best, args)
        end_time = time.time()
        epoch_time = end_time - start_time
        print("An Epoch Time: ", epoch_time)
        txt_name = checkpoint_path + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(str(epoch_time) + '\n')

def train(train_loader, model, criterion, criterion_af, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):

        images = images.cuda()
        target = target.cuda()

        # compute output
        output, output2 = model(images)
        loss = (args.beta * criterion(output, target)) + ((1-args.beta) * criterion_af(output2, target))

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)
            
    # tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss, optimizer.param_groups[0]['lr']))
    return top1.avg, losses.avg

def validate(val_loader, model, criterion, criterion_af, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output, output2 = model(images)
            loss = (args.beta * criterion(output, target)) + ((1 - args.beta) * criterion_af(output2, target))

            # measure accuracy and record loss
            acc, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

        print(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))
        with open(checkpoint_path + time_str + 'log.txt', 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')
    return top1.avg, losses.avg

if __name__ == '__main__':
    main()