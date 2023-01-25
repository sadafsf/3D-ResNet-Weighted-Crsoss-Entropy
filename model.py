import torch
from torch import nn

import resnet


def generate_model(args):
    assert args.mode in ['score', 'feature']
    if args.mode == 'score':
        last_fc = True
    elif args.mode == 'feature':
        last_fc = False

    if args.model_depth == 10:
        model = resnet.resnet10(num_classes=args.n_classes, shortcut_type=args.resnet_shortcut,
                                sample_size=args.sample_size, sample_duration=args.sample_duration,
                                last_fc=last_fc)
    elif args.model_depth == 18:
        model = resnet.resnet18(num_classes=args.n_classes, shortcut_type=args.resnet_shortcut,
                                sample_size=args.sample_size, sample_duration=args.sample_duration,
                                last_fc=last_fc)

    elif args.model_depth == 50:
        model = resnet.resnet50(num_classes=args.n_classes, shortcut_type=args.resnet_shortcut,
                                sample_size=args.sample_size, sample_duration=args.sample_duration,
                                last_fc=last_fc)
    elif args.model_depth == 101:
        model = resnet.resnet101(num_classes=args.n_classes, shortcut_type=args.resnet_shortcut,
                                sample_size=args.sample_size, sample_duration=args.sample_duration,
                                last_fc=last_fc)



   
    model = nn.DataParallel(model, device_ids=None)
    if args.use_cuda:
        model.cuda()

    return model