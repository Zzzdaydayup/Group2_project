# 重新训练 -->python train.py --data /path/to/traffic_sign.yaml --weights yolov5s.pt --epochs 50 --batch-size 16 --img-size 640 640


import torch
from torch.utils.tensorboard import SummaryWriter
from utils.general import increment_path, check_img_size, set_logging, check_requirements
from utils.torch_utils import select_device, time_synchronized
from utils.plots import plot_results, plot_confusion_matrix
from utils.metrics import ConfusionMatrix
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.loss import ComputeLoss
from utils.general import fitness, labels_to_class_weights, labels_to_image_weights
import yaml
import os
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/traffic_sign.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train, val image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload-dataset', action='store_true', help='Upload dataset to W&B')
    parser.add_argument('--bbox-interval', type=int, default=-1, help='set bounding-box image logging interval')
    parser.add_argument('--save-period', type=int, default=-1, help='Log model after every "save_period" epochs')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    opt = parser.parse_args()
    return opt

def main(opt):
    set_logging()
    device = select_device(opt.device)
    if device.type == 'cpu':
        mixed_precision = False

    save_dir = increment_path(os.path.join(opt.project, opt.name), exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if opt.evolve else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)
    nc = int(data_dict['nc'])  # number of classes

    # Initialize model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    gs = int(model.stride.max())  # grid size (max stride)
    imgsz = check_img_size(opt.img_size[0], gs)  # check img_size

    # Training
    hyp = opt.hyp
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.names = data_dict['names']  # attach class names to model

    # Dataloader
    dataloader, dataset = create_dataloader(opt.data, imgsz, opt.batch_size, gs, opt.rect, opt.cache_images, opt.single_cls, opt.pad, opt.rect)
    mloss = torch.zeros(3, device=device)  # mean losses
    nb = len(dataloader)  # number of batches

    # Define loss function
    compute_loss = ComputeLoss(model)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / opt.batch_size), 1)  # accumulate loss before optimizing
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        if 'bias' in k:
            pg2 += [v]  # biases
        elif 'weight' in k and '.bn' not in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if opt.adam:
        optimizer = torch.optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = torch.optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    lf = lambda x: (1 - x / opt.epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Start training
    for epoch in range(opt.epochs):  # epoch ------------------------------------------------------------------
        model.train()
        for i, (imgs, targets, paths, _) in enumerate(dataloader):
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)

            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets)
            loss.backward()

            if i % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            mloss = (mloss * i + loss_items) / (i + 1)

        scheduler.step()

        # Validation
        if not opt.notest or (epoch + 1) == opt.epochs:
            results, maps, times = test(data_dict, batch_size=opt.batch_size, imgsz=imgsz, model=model, conf_thres=0.001)

        # Update results
        if opt.bucket:
            os.system('gsutil -m cp -r runs/train/exp gs://%s' % opt.bucket)

        plot_results(save_dir=save_dir)

    print('Finished training.')

    # Evaluate with confusion matrix
    confmat = ConfusionMatrix(nc=nc)
    for i, (imgs, targets, paths, shapes) in enumerate(dataloader):
        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        targets = targets.to(device)

        with torch.no_grad():
            pred = model(imgs)

        confmat.process_batch(pred, targets)
    
    plot_confusion_matrix(confmat.matrix, save_dir=save_dir, names=data_dict['names'])
    print('Confusion matrix saved.')

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
