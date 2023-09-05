from rl_model.config import args
from rl_model.config.mode import Mode
from rl_model.rl import RL_Experiment
from config import Configuration


if __name__ == "__main__":
    experiment = RL_Experiment(Mode.TRAIN, args.config(mode=Mode.TRAIN),Configuration().get_seed())
    experiment()