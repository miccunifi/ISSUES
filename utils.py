import argparse


def str2bool(v):
    """
    src: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def generate_name(args):
    terms = [args.name, args.dataset]

    if args.name in ['combiner', 'text-inv-comb']:
        if getattr(args, 'pretrained_proj_weights'):
            if getattr(args, 'freeze_proj_layers'):
                terms.append('two-stage-pretrain')
            else:
                terms.append('two-stage-finetune')

    if args.name in ['sum', 'combiner']:
        terms.append(f'{args.num_mapping_layers}-{args.map_dim}-proj-map')

    if args.name in ['text-only', 'image-only']:
        if getattr(args, 'proj_map'):
            terms.append(f'{args.num_mapping_layers}-{args.map_dim}-proj-map')

    if args.name in ['text-inv', 'text-inv-fusion', 'text-inv-comb']:
        if getattr(args, 'text_inv_proj'):
            terms.append('clip-text-proj')
        if getattr(args, 'phi_inv_proj'):
            terms.append('phi-proj')
        if getattr(args, 'post_inv_proj'):
            terms.append('post-proj')
        if getattr(args, 'enh_text'):
            terms.append('enh-text')
        if getattr(args, 'phi_freeze'):
            terms.append('phi-frozen')
        else:
            terms.append('phi-finetune')

    if args.name in ['adaptation', 'hate-clipper', 'text-inv-fusion']:
        terms.append(args.fusion)

    if args.name in ['combiner', 'text-inv-comb']:
        if getattr(args, 'comb_proj'):
            terms.append('comb-proj')
        terms.append(f"{getattr(args, 'comb_fusion')}-comb")
        if getattr(args, 'convex_tensor'):
            terms.append('convex_tensor')

    terms.append(f'{args.num_pre_output_layers}-MLP')

    if getattr(args, 'fast_process'):
        terms.append('fast-mode')

    return '_'.join([elem for elem in terms])
