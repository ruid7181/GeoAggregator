import time
import argparse
from model.aggregator_train import train_aggregator


def main():
    parser = argparse.ArgumentParser(description="Train aggregator model with specified settings.")

    # Select ga model variants...
    parser.add_argument("--variant", type=str, choices=["mini", "small", "large"],
                        help="Select the model variant: mini, small, or large.")
    # ... or specify hyperparameters manually.
    parser.add_argument("--lyr", type=int,
                        help="Number of attention layers. Required if variant is not provided.")
    parser.add_argument("--sequence_len", type=int,
                        help="Sequence length. Required if variant is not provided.")
    parser.add_argument("--inducing_points", type=int,
                        help="Number of inducing points. Required if variant is not provided.")

    parser.add_argument("--epoch", type=int, default=30,
                        help="Number of epochs. Required if variant is not provided.")
    parser.add_argument("--radius", type=float, default=None,
                        help="Radius of ContextQuery operation.")

    parser.add_argument("--dataset", type=str, default="syn-durbin-d",
                        help="The dataset name, e.g. syn-durbin-d")
    parser.add_argument("--not_decreasing_rounds", type=int, default=100,
                        help="Number of not decreasing rounds for early stopping.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training.")
    parser.add_argument("--attn_bias_factor", type=float, default=None,
                        help="Attention bias factor. Use None for learnable bias factor.")
    parser.add_argument("--model_save_fn", type=str, default=None,
                        help="File name to save the model. If None, the model won't be saved.")

    args = parser.parse_args()

    predefined_variants = {
        'mini':  {'lyr': 1, 'sl': 81,  'inducing_points': 1},
        'small': {'lyr': 2, 'sl': 144, 'inducing_points': 4},
        'large': {'lyr': 3, 'sl': 256, 'inducing_points': 8}
    }

    if args.variant:
        if args.lyr is not None or args.sequence_len is not None or args.inducing_points is not None:
            parser.error("Cannot specify --epoch, --sequence_len, or --lyr when --variant is provided.")

        chosen_variant = predefined_variants[args.variant]
        epoch = args.epoch
        lyr = chosen_variant['lyr']
        sequence_len = chosen_variant['sl']
        inducing_points = chosen_variant['inducing_points']

        print(f'Epoch = {epoch}, '
              f'model variant = {args.variant}: {chosen_variant}, '
              f'radius = {args.radius}, '
              f'attn_bias_factor = {args.attn_bias_factor}, '
              f'dataset = {args.dataset}.')

    else:
        if args.epoch is None or args.sequence_len is None or args.lyr is None:
            parser.error("When --variant is not provided, --epoch, --sequence_len, and --lyr are required.")
        epoch = args.epoch
        lyr = args.lyr
        sequence_len = args.sequence_len
        inducing_points = args.inducing_points

        print(f'Epoch = {epoch}, '
              f'attn_lyr = {lyr}, '
              f'seq_len = {sequence_len}, '
              f'inducing_points = {inducing_points}, '
              f'radius = {args.radius}, '
              f'attn_bias_factor = {args.attn_bias_factor}, '
              f'dateset = {args.dataset}.')

    begin_time = time.time()

    train_aggregator(
        dataset=args.dataset,
        epoch=epoch,
        not_decreasing_rounds=args.not_decreasing_rounds,
        sample_radius=args.radius,
        sequence_len=sequence_len,
        batch_size=args.batch_size,
        n_attn_layer=lyr,
        inducing_points=inducing_points,
        attn_bias_factor=args.attn_bias_factor,
        model_save_fn=args.model_save_fn
    )

    print(f'Elapsed time: {(time.time() - begin_time) / 3600} h.')


if __name__ == "__main__":
    main()
