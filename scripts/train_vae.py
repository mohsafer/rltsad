"""Train the reconstruction model on normal multivariate windows
(Algorithm 1, BUILDVAE).

The backbone is pluggable:

* ``--model vae`` (default) -- the paper's Variational Autoencoder
  (Sec. III-B / IV-A): ELBO loss = reconstruction + KL;
* ``--model transformer`` -- the Transformer autoencoder of
  ``models/transformer/`` (per-time-step tokens, self-attention encoder,
  10-d latent bottleneck, sigmoid decoder); with ``--variational`` it keeps
  the VAE's ELBO objective, otherwise it trains with pure reconstruction
  loss.

Both produce the identical artifact consumed by the rest of the pipeline
(Algorithm 1 COMPUTEPENALTY): per-window reconstruction error as the
unsupervised anomaly score / intrinsic reward R2.

Steps (paper Sec. IV-A):

1. load the dataset, remove sensors with zero variance across the training
   samples and Min-Max normalise (identical to the RL side);
2. slide length-``n_steps`` windows over normal segments only, scale and
   flatten each window to a vector of size n_steps * d;
3. train the backbone to minimise its reconstruction (+ KL) loss;
4. save weights, scaler and metadata under ``models/<model>/<run_name>/``.

Example (synthetic data):
    python scripts/train_vae.py --model transformer \
        --dataset synthetic --data_dir data/synthetic \
        --output_dir models/transformer --run_name synthetic
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from models.reconstruction import (  # noqa: E402
    reconstruction_errors,
    save_reconstructor,
    train_reconstructor,
)
from models.transformer.transformer_model import TransformerAE  # noqa: E402
from models.vae.vae_model import VAE  # noqa: E402
from utils.config import DRSMTConfig, set_seed  # noqa: E402
from utils.data_loader import (  # noqa: E402
    collect_normal_windows,
    flatten_windows,
    make_windows,
    normalize_data,
    prepare_data,
    window_labels,
)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the DRSMT reconstruction model (BUILDVAE): "
        "VAE (paper) or Transformer autoencoder."
    )
    parser.add_argument("--model", default="vae", choices=["vae", "transformer"],
                        help="reconstruction backbone (default: vae = the paper's model)")
    parser.add_argument("--dataset", default="synthetic",
                        choices=["smd", "synthetic", "wadi", "yahoo"])
    parser.add_argument("--data_dir", default="data/synthetic")
    parser.add_argument("--output_dir", default=None,
                        help="default: models/<model>")
    parser.add_argument("--run_name", default=None,
                        help="sub-directory of --output_dir (default: --dataset)")
    parser.add_argument("--n_steps", type=int, default=25)
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--scaler", default="standard", choices=["standard", "robust"],
                        help="Algorithm 1 line 6 'Standardize' (default); "
                             "'robust' reproduces the WADI variant")
    parser.add_argument("--scale_clip", type=float, default=10.0)
    parser.add_argument("--max_samples", type=int, default=100_000)
    parser.add_argument("--no_drop_zero_variance", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    vae_g = parser.add_argument_group("VAE backbone (--model vae)")
    vae_g.add_argument("--intermediate_dim", type=int, default=64)
    vae_g.add_argument("--encoder_layers", type=int, default=3)

    tr_g = parser.add_argument_group("Transformer backbone (--model transformer)")
    tr_g.add_argument("--d_model", type=int, default=128)
    tr_g.add_argument("--nhead", type=int, default=4)
    tr_g.add_argument("--enc_layers", type=int, default=2,
                      help="transformer encoder blocks")
    tr_g.add_argument("--dec_layers", type=int, default=2,
                      help="transformer decoder blocks")
    tr_g.add_argument("--ff_dim", type=int, default=256)
    tr_g.add_argument("--dropout", type=float, default=0.1)
    tr_g.add_argument("--variational", action="store_true",
                      help="Transformer-VAE: keep the paper's ELBO (recon + KL) objective")
    return parser.parse_args(argv)


def build_reconstruction_model(args, input_dim: int, n_features: int):
    """Build the selected reconstruction backbone."""
    if args.model == "vae":
        return VAE(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            intermediate_dim=args.intermediate_dim,
            encoder_layers=args.encoder_layers,
        )
    return TransformerAE(
        input_dim=input_dim,
        n_steps=args.n_steps,
        n_features=n_features,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ff_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        variational=args.variational,
    )


def main(argv=None) -> None:
    args = _parse_args(argv)
    cfg = DRSMTConfig(
        n_steps=args.n_steps,
        recon_model=args.model,
        vae_latent_dim=args.latent_dim,
        vae_intermediate_dim=args.intermediate_dim,
        vae_encoder_layers=args.encoder_layers,
        vae_epochs=args.epochs,
        vae_batch_size=args.batch_size,
        vae_learning_rate=args.learning_rate,
        vae_scaler=args.scaler,
        vae_scale_clip=args.scale_clip,
        max_vae_samples=args.max_samples,
        drop_zero_variance=not args.no_drop_zero_variance,
        seed=args.seed,
        device=args.device,
        transformer_d_model=args.d_model,
        transformer_nhead=args.nhead,
        transformer_num_encoder_layers=args.enc_layers,
        transformer_num_decoder_layers=args.dec_layers,
        transformer_dim_feedforward=args.ff_dim,
        transformer_dropout=args.dropout,
        transformer_variational=args.variational,
    )
    set_seed(cfg.seed)
    device = cfg.resolved_device()
    print(f"[train-{args.model}] device: {device}")

    # ---------------------------------------------------------------- data
    # Normalise exactly like the RL side (scripts.common.load_and_prepare):
    # Min-Max statistics are fitted on the training series, and the
    # WindowScaler below is then fitted on the *same normalised* windows that
    # COMPUTEPENALTY feeds the model during warm-up / training / evaluation.
    data = normalize_data(
        prepare_data(args.dataset, args.data_dir, cfg.drop_zero_variance)
    )
    n_features = data.n_features
    input_dim = cfg.n_steps * n_features
    print(
        f"[train-{args.model}] dataset '{data.dataset}': {len(data.train)} train / "
        f"{len(data.test)} test series, d={n_features} features "
        f"(after zero-variance removal), input_dim={input_dim}"
    )

    windows = collect_normal_windows(
        data.train, cfg.n_steps, cfg.max_vae_samples,
        rng=np.random.default_rng(cfg.seed),
    )
    print(f"[train-{args.model}] collected {windows.shape[0]} fully-normal windows "
          f"of length {cfg.n_steps}")

    # ------------------------------------------------------------- scaling
    from models.vae.vae_model import WindowScaler

    scaler = WindowScaler(kind=cfg.vae_scaler, clip=cfg.vae_scale_clip)
    x_scaled = scaler.fit_transform(windows)

    # ---------------------------------------------------------------- model
    model = build_reconstruction_model(args, input_dim, n_features)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train-{args.model}] model parameters: {n_params:,}")
    losses = train_reconstructor(
        model,
        x_scaled,
        epochs=cfg.vae_epochs,
        batch_size=cfg.vae_batch_size,
        learning_rate=cfg.vae_learning_rate,
        grad_clip=cfg.vae_grad_clip,
        device=device,
    )

    # --------------------------------------------------- sanity check report
    sample = x_scaled[: min(2_000, len(x_scaled))]
    train_err = reconstruction_errors(model, sample, device=device)
    print(
        f"[train-{args.model}] reconstruction MSE on normal windows: "
        f"mean={train_err.mean():.6f} median={np.median(train_err):.6f} "
        f"p95={np.percentile(train_err, 95):.6f}"
    )
    if len(data.test) and (data.test[0].labels == 1).any():
        series = data.test[0]
        test_windows = make_windows(series.values, cfg.n_steps)
        if len(test_windows) > 50_000:  # keep the report lightweight
            sel = np.random.default_rng(cfg.seed).choice(
                len(test_windows), size=50_000, replace=False
            )
            test_windows = test_windows[np.sort(sel)]
        test_flat = scaler.transform(flatten_windows(test_windows))
        test_err = reconstruction_errors(model, test_flat, device=device)
        test_labels = window_labels(series.labels, cfg.n_steps)
        normal_err = test_err[test_labels == 0]
        anom_err = test_err[test_labels == 1]
        if len(normal_err) and len(anom_err):
            print(
                f"[train-{args.model}] separability on '{series.name}': "
                f"mean error normal={normal_err.mean():.6f} "
                f"vs anomaly={anom_err.mean():.6f} "
                f"(ratio={anom_err.mean() / max(normal_err.mean(), 1e-12):.2f}x)"
            )

    # ----------------------------------------------------------------- save
    architecture = "vae" if args.model == "vae" else "transformer_ae"
    out_dir = os.path.join(args.output_dir or f"models/{args.model}",
                           args.run_name or data.dataset)
    meta = {
        "architecture": architecture,
        "backbone_flag": args.model,
        "dataset": data.dataset,
        "data_dir": args.data_dir,
        "n_steps": cfg.n_steps,
        "n_features": n_features,
        "input_dim": input_dim,
        "feature_names": data.feature_names,
        "scaler": cfg.vae_scaler,
        "scale_clip": cfg.vae_scale_clip,
        "latent_dim": cfg.vae_latent_dim,
        "epochs": cfg.vae_epochs,
        "drop_zero_variance": cfg.drop_zero_variance,
        "train_windows": int(windows.shape[0]),
        "final_loss": float(losses[-1]) if losses else None,
        "intermediate_dim": args.intermediate_dim if args.model == "vae" else None,
        "encoder_layers": args.encoder_layers if args.model == "vae" else None,
        "d_model": args.d_model if args.model == "transformer" else None,
        "nhead": args.nhead if args.model == "transformer" else None,
        "num_encoder_layers": args.enc_layers if args.model == "transformer" else None,
        "num_decoder_layers": args.dec_layers if args.model == "transformer" else None,
        "dim_feedforward": args.ff_dim if args.model == "transformer" else None,
        "dropout": args.dropout if args.model == "transformer" else None,
        "variational": bool(args.variational) if args.model == "transformer" else True,
    }
    meta = {k: v for k, v in meta.items() if v is not None}
    save_reconstructor(out_dir, model, scaler, meta)
    print(f"[train-{args.model}] saved {architecture} weights, scaler and meta "
          f"to {out_dir}")


if __name__ == "__main__":
    main()
