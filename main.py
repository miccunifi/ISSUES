import argparse
import random
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from datasets import MemesCollator, load_dataset
from engine import create_model, HateClassifier
from utils import str2bool, generate_name


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Training and evaluation script for hateful memes classification')

    parser.add_argument('--dataset', default='hmc', choices=['hmc', 'harmeme'])
    parser.add_argument('--image_size', type=int, default=224)

    parser.add_argument('--num_mapping_layers', default=1, type=int)
    parser.add_argument('--map_dim', default=768, type=int)

    parser.add_argument('--fusion', default='align',
                        choices=['align', 'concat'])

    parser.add_argument('--num_pre_output_layers', default=1, type=int)

    parser.add_argument('--drop_probs', type=float, nargs=3, default=[0.1, 0.4, 0.2],
                        help="Set drop probabilities for map, fusion, pre_output")

    parser.add_argument('--gpus', default='0', help='GPU ids concatenated with space')
    parser.add_argument('--limit_train_batches', default=1.0)
    parser.add_argument('--limit_val_batches', default=1.0)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--max_epochs', type=int, default=-1)
    parser.add_argument('--log_every_n_steps', type=int, default=25)
    parser.add_argument('--val_check_interval', default=1.0)
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gradient_clip_val', type=float, default=0.1)

    parser.add_argument('--proj_map', default=False, type=str2bool)

    parser.add_argument('--pretrained_proj_weights', default=False, type=str2bool)
    parser.add_argument('--freeze_proj_layers', default=False, type=str2bool)

    parser.add_argument('--comb_proj', default=False, type=str2bool)
    parser.add_argument('--comb_fusion', default='align',
                        choices=['concat', 'align'])
    parser.add_argument('--convex_tensor', default=False, type=str2bool)

    parser.add_argument('--text_inv_proj', default=False, type=str2bool)
    parser.add_argument('--phi_inv_proj', default=False, type=str2bool)
    parser.add_argument('--post_inv_proj', default=False, type=str2bool)

    parser.add_argument('--enh_text', default=False, type=str2bool)

    parser.add_argument('--phi_freeze', default=False, type=str2bool)

    parser.add_argument('--name', type=str, default='adaptation',
                        choices=['adaptation', 'hate-clipper', 'image-only', 'text-only', 'sum', 'combiner', 'text-inv',
                                 'text-inv-fusion', 'text-inv-comb']
                        )
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--reproduce', default=False, type=str2bool)
    parser.add_argument('--print_model', default=False, type=str2bool)
    parser.add_argument('--fast_process', default=False, type=str2bool)

    return parser


def main(args):
    run_name = f'{generate_name(args)}-{random.randint(0, 1000000000)}'

    seed_everything(42, workers=True)

    # load dataset
    if args.dataset == 'hmc':
        dataset_train = load_dataset(args=args, split='train')
        dataset_val = load_dataset(args=args, split='dev_seen')
        dataset_test = load_dataset(args=args, split='test_seen')
        dataset_val_unseen = load_dataset(args=args, split='dev_unseen')
        dataset_test_unseen = load_dataset(args=args, split='test_unseen')

    elif args.dataset == 'harmeme':
        dataset_train = load_dataset(args=args, split='train')
        dataset_val = load_dataset(args=args, split='val')
        dataset_test = load_dataset(args=args, split='test')

    else:
        raise ValueError()

    print("Number of training examples:", len(dataset_train))
    print("Number of validation examples:", len(dataset_val))
    print("Number of test examples:", len(dataset_test))
    if args.dataset == 'hmc':
        print("Number of validation examples (unseen):", len(dataset_val_unseen))
        print("Number of test examples (unseen):", len(dataset_test_unseen))

    # data loader
    num_cpus = 0 if args.fast_process else min(args.batch_size, 24)

    collator = MemesCollator(args)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collator, num_workers=num_cpus)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=collator, num_workers=num_cpus)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,
                                 collate_fn=collator, num_workers=num_cpus)

    if args.dataset == 'hmc':
        dataloader_val_unseen = DataLoader(dataset_val_unseen, batch_size=args.batch_size,
                                           collate_fn=collator, num_workers=num_cpus)
        dataloader_test_unseen = DataLoader(dataset_test_unseen, batch_size=args.batch_size,
                                            collate_fn=collator, num_workers=num_cpus)

    if args.reproduce:
        assert args.pretrained_model.startswith(f'{args.dataset}'), 'The pretrained model and the dataset selected' \
                                                                   ' are not compatible'
        # load pretrained model
        model = HateClassifier.load_from_checkpoint(f'resources/pretrained_models/{args.pretrained_model}', args=args,
                                                    map_location=None)
    else:
        # create model
        model = create_model(args)

    if args.print_model:
        print(model)

        # print(model.state_dict())

        # for name, p in model.named_parameters():
        #     print('{}: {}'.format(name,p.requires_grad))
        return

    project = "hateful-memes"

    wandb_logger = WandbLogger(project=project, name=run_name, config=args)

    num_params = {f'param_{n}': p.numel() for n, p in model.named_parameters() if p.requires_grad}
    wandb_logger.experiment.config.update(num_params)

    monitor = "val/auroc"
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints', filename=run_name+'-{epoch:02d}',
                                          monitor=monitor, mode='max', verbose=True, save_weights_only=True,
                                          save_top_k=1, save_last=False)

    trainer = Trainer(accelerator='gpu', devices=args.gpus, max_epochs=args.max_epochs, max_steps=args.max_steps,
                      gradient_clip_val=args.gradient_clip_val, logger=wandb_logger,
                      log_every_n_steps=args.log_every_n_steps, val_check_interval=args.val_check_interval,
                      callbacks=[checkpoint_callback], limit_train_batches=args.limit_train_batches,
                      limit_val_batches=args.limit_val_batches, deterministic=True)

    if not args.reproduce:
        trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

        if args.dataset == 'hmc':
            trainer.test(ckpt_path='best',
                         dataloaders=[dataloader_val, dataloader_test, dataloader_val_unseen, dataloader_test_unseen]
                         )
        elif args.dataset == 'harmeme':
            trainer.test(ckpt_path='best',
                         dataloaders=[dataloader_val, dataloader_test]
                         )
        else:
            raise ValueError()
    else:
        if args.dataset == 'hmc':
            trainer.test(model,
                         dataloaders=[dataloader_val, dataloader_test, dataloader_val_unseen, dataloader_test_unseen]
                         )
        elif args.dataset == 'harmeme':
            trainer.test(model,
                         dataloaders=[dataloader_val, dataloader_test]
                         )
        else:
            raise ValueError()


if __name__ == '__main__':
    pars = get_arg_parser()
    arguments = pars.parse_args()
    arguments.gpus = [int(id_) for id_ in arguments.gpus.split()]
    for i in arguments.gpus:
        print('current device: {}'.format(torch.cuda.get_device_properties(i)))

    main(arguments)
