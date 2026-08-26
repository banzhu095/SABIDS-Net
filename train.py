from __future__ import annotations

import argparse

from sabids.config import load_config, save_config
from sabids.engine import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SABIDS-Net")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trainer = Trainer(config)
    save_config(config, trainer.output_dir / "resolved_config.yaml")
    trainer.fit()


if __name__ == "__main__":
    main()

