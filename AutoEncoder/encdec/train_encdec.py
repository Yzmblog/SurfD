import sys

sys.path.append("..")

from hesiod import hmain

from trainers.encdec import EncoderDecoderTrainer



if len(sys.argv) > 1:
    run_cfg_file = sys.argv[1]
    del sys.argv[1]
else:
    run_cfg_file = None


@hmain(
    base_cfg_dir="../cfg/bases",
    template_cfg_file="../cfg/encdec.yaml",
    run_cfg_file=run_cfg_file,
    out_dir_root="../logs",
)
def main() -> None:
    trainer = EncoderDecoderTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
