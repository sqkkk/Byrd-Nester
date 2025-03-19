import os
import subprocess
import argparse
from utils import get_args
from utils import main
from utils import EXP_DIR

def get_args():
    parser = argparse.ArgumentParser(description="")

    # Utility
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--identifier", type=str, default="debug", help="")
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="If plot is enabled, then ignore all other options.",
    )

    # Experiment configuration
    parser.add_argument("-n", type=int, default=25, help="Number of workers")
    parser.add_argument("-f", type=int, default=5, help="Number of Byzantine workers.")
    parser.add_argument("--attack", type=str, default="IPM", help="Type of attacks.")
    parser.add_argument("--agg", type=str, default="rfa", help="")
    parser.add_argument(
        "--noniid",
        action="store_true",
        default=True,
        help="[HP] noniidness.",
    )
    parser.add_argument("--LT", action="store_true", default=False, help="Long tail")

    # Key hyperparameter
    parser.add_argument("--bucketing", type=int, default=0, help="[HP] s")
    parser.add_argument("--momentum", type=float, default=0.9, help="[HP] momentum")
    parser.add_argument("--nesterov", action="store_true", default=False)

    parser.add_argument("--clip-tau", type=float, default=10.0, help="[HP] momentum")
    parser.add_argument("--clip-scaling", type=str, default=None, help="[HP] momentum")

    parser.add_argument(
        "--mimic-warmup", type=int, default=1, help="the warmup phase in iterations."
    )

    parser.add_argument(
        "--op",
        type=int,
        default=1,
        help="[HP] controlling the degree of overparameterization. "
             "Only used in exp8.",
    )

    args = parser.parse_args()

    if args.n <= 0 or args.f < 0 or args.f >= args.n:
        raise RuntimeError(f"n={args.n} f={args.f}")

    assert args.bucketing >= 0, args.bucketing
    assert args.momentum >= 0, args.momentum
    assert len(args.identifier) > 0
    return args

def run_exp():
    COMMON_OPTIONS = ["--use-cuda", "--identifier", "all", "-n", "25", "-f", "5", "--noniid"]
    processes = []

    for atk in ["BF", "LF", "mimic", "IPM", "ALIE"]:
        for s in [0, 2]:
            args_list = COMMON_OPTIONS + ["--attack", atk, "--agg", "cp", "--bucketing", str(s), "--seed", "0", "--momentum", "0"]
            processes.append(subprocess.Popen(["/home/sqk/miniconda3/envs/federated-env/bin/python", "exp.py"] + args_list))

            for m in [0.5, 0.9, 0.99]:
                args_list = COMMON_OPTIONS + ["--attack", atk, "--agg", "cp", "--bucketing", str(s), "--seed", "0", "--momentum", str(m)]
                processes.append(subprocess.Popen(["/home/sqk/miniconda3/envs/federated-env/bin/python", "exp.py"] + args_list))

                args_list = COMMON_OPTIONS + ["--attack", atk, "--agg", "cp", "--bucketing", str(s), "--seed", "0", "--momentum", str(m), "--clip-scaling", "linear"]
                processes.append(subprocess.Popen(["/home/sqk/miniconda3/envs/federated-env/bin/python", "exp.py"] + args_list))

                args_list = COMMON_OPTIONS + ["--attack", atk, "--agg", "cp", "--bucketing", str(s), "--seed", "0", "--momentum", str(m), "--clip-scaling", "sqrt"]
                processes.append(subprocess.Popen(["/home/sqk/miniconda3/envs/federated-env/bin/python", "exp.py"] + args_list))

    for p in processes:
        p.wait()

def main_experiment(args):
    assert args.noniid
    assert not args.LT
    # assert args.agg == "rfa"

    LOG_DIR = EXP_DIR + "exp/"
    if args.identifier:
        LOG_DIR += f"{args.identifier}/"
    elif args.debug:
        LOG_DIR += "debug/"
    else:
        LOG_DIR += f"n{args.n}_f{args.f}_{args.agg}_{args.noniid}/"

    INP_DIR = LOG_DIR
    OUT_DIR = LOG_DIR + "output/"
    LOG_DIR += f"{args.attack}_{args.agg}_{args.noniid}_{args.momentum}_{args.nesterov}_s{args.bucketing}_{args.clip_scaling}_seed{args.seed}"

    if args.debug:
        MAX_BATCHES_PER_EPOCH = 30
        EPOCHS = 3
    else:
        MAX_BATCHES_PER_EPOCH = 30
        EPOCHS = 40

    if not args.plot:
        main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH)
    else:
        # Temporarily put the import functions here to avoid
        # random error stops the running processes.
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from codes.parser import extract_validation_entries

        def exp_grid():
            for attack in ["ALIE", "SF"]:
                for bucketing in [2]:
                    for momentum in [0,0.9]:
                        for scaling in ["None"]:
                            for agg in ["rfa"]:
                                yield attack, momentum, bucketing, scaling, agg

        results = []
        for attack, momentum, bucketing, scaling, agg in exp_grid():
            args.attack = attack
            args.agg = agg
            args.momentum = momentum
            args.nesterov = False
            args.bucketing = bucketing
            args.clip_scaling = scaling
            grid_identifier = f"{args.attack}_{args.agg}_{args.noniid}_{args.momentum}_{args.nesterov}_s{args.bucketing}_{args.clip_scaling}_seed0"
            path = INP_DIR + grid_identifier + "/stats"
            try:
                values = extract_validation_entries(path)
                i = 0
                for v in values:
                    results.append(
                        {
                            "Iterations": v["E"] * MAX_BATCHES_PER_EPOCH,
                            "Accuracy (%)": v["top1"],
                            "ATK": attack,
                            r"$\beta$": str(momentum),
                            "Scaling": scaling if scaling != "None" else "NA",
                            "Bucketing": str(bucketing),
                        }
                    )
                    i = i + 1
                    print(i,v["top1"])
            except Exception as e:
                pass

        # results = pd.DataFrame(results)
        # print(results)
        #
        # if not os.path.exists(OUT_DIR):
        #     os.makedirs(OUT_DIR)
        #
        # sns.set(font_scale=1.25)
        # g = sns.relplot(
        #     data=results,
        #     x="Iterations",
        #     y="Accuracy (%)",
        #     col="ATK",
        #     row="Scaling",
        #     style="Bucketing",
        #     # hue="Resampling",
        #     hue=r"$\beta$",
        #     kind="line",
        #     ci=None,
        #     height=2.5,
        #     aspect=1.3,
        # )
        # g.set(xlim=(0, 1200), ylim=(0, 100))
        # g.fig.savefig(OUT_DIR + "exp.pdf", bbox_inches="tight")

if __name__ == "__main__":
    args = get_args()

    if args.identifier == 'all':
        run_exp()
    else:
        # main_experiment(args)
        for agg in ["tm"]:
            for atk in ["BF","LF","IPM"]:
                for m in [0,0.9]:
                    args.agg = agg
                    args.attack = atk
                    args.momentum = m
                    args.nesterov = False
                    main_experiment(args)
