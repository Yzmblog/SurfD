from argparse import ArgumentParser
import argparse
import os
import json


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--num_actions", default=9, type=int, help="num_classes.")
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
    group.add_argument("--distributed", default=False, type=bool, help="Use ddp to train model")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='OpenUNet',
                       choices=['OpenUNet'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--cond_mask_prob", default=0, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action.")
    group.add_argument("--cond_mode",
                       choices=['no_cond', 'text', 'sketch', 'category', 'img'], type=str,required=True,
                       help="condition type")


def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='deepfashion3d', choices=['deepfashion3d', 'text2shape', 'pix3d', 'kcars'
                                                                ], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")

    group.add_argument("--ae_dir", required=False, type=str,
                       help="Path to save checkpoints and results.")

    group.add_argument("--num_workers", default=4, type=int, help="num_workers.")

    group.add_argument("--grid_size", default=128, type=int, help="grid size.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")

    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")

    group.add_argument("--log_interval", default=10, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=50_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--clip_value", default=0.1, type=float, help="max_clipping value (0-max).")
    group.add_argument("--guidance_param", default=1.0, type=float, help="Classifier Free Guidance")#3.0


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=1, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--guidance_param", default=1.0, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--if_clip", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--clip_value", default=0.1, type=float, help="max_clipping value (0-max).")



def add_generate_options(parser):
    group = parser.add_argument_group('generate')

    group.add_argument("--grid_size", default=128, type=int, help="grid size.")
    group.add_argument("--category", default=0, type=int, required=False,
                        help="Condition category.")
    group.add_argument("--sketch_path", default=None, type=str, required=False,
                       help="Path to the condition sketch image.")
    group.add_argument("--image_path", default=None, type=str, required=False,
                       help="Path to the condition image.")
    group.add_argument("--mask_path", default=None, type=str, required=False,
                       help="Path to the condition mask.")
    group.add_argument("--prompt", default=None, type=str, required=False,
                       help="text prompt for generation.")
    group.add_argument("--watertight",  action='store_true',
                       help="mesh attributes.")
    group.add_argument("--resolution",  default=512, type=int, required=False,
                       help="mesh resolution.")
    group.add_argument("--ae_dir", default=None, type=str, help="Path to ae")



def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    parser.add_argument("--local_rank", type=int)
    return parser.parse_args()


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    return parse_and_load_from_model(parser)