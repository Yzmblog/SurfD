from models.mdm import MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps


def load_model_wo_clip(model, state_dict):
    missing_keys, _ = model.load_state_dict(state_dict, strict=False)

    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model_and_diffusion(args):
    model = MDM(**get_model_args(args))

    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args):
    '''
    we do not set action number here.
    '''
    # default args
    clip_version = 'ViT-B/32'
    cond_mode = args.cond_mode

    return {'modeltype': '', 'num_actions': args.num_actions,
            'dropout': 0.1, 'activation': "gelu", 'cond_mode': cond_mode,
            'arch': args.arch, 'clip_version': clip_version, 'dataset': args.dataset}


def create_gaussian_diffusion(args):
    #
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!

    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        args = args,
    )