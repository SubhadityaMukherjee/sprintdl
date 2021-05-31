from ..callbacks import *


class AvgGANStats:
    """
    Store statistics of training
    """

    def __init__(self, metrics, in_train):
        self.metrics, self.in_train = listify(metrics), in_train

    def reset(self):
        self.tot_loss_G, self.tot_loss_D, self.count = 0.0, 0.0, 1
        self.tot_mets = [0.0] * len(self.metrics)

    @property
    def all_stats(self):
        return [self.tot_loss_G.item(), self.tot_loss_D.item()] + self.tot_mets

    @property
    def avg_stats(self):
        return [o / self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count:
            return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):
        bn = run.xb.shape[0]
        #         print(run.loss_D)
        self.tot_loss_G += run.loss_G.to(run.device) * bn
        self.tot_loss_D += run.loss_D.to(run.device) * bn
        self.count += bn
        for i, m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn


class AvgGANStatsCallback(Callback):
    """
    Main callback for stats for GANS Used for the progress bar
    """

    def __init__(self, metrics):
        self.train_stats, self.valid_stats = AvgGANStats(metrics, True), AvgGANStats(
            metrics, False
        )

    def begin_fit(self):
        met_names = ["loss_D", "loss_G"] + [
            m.__name__ for m in self.train_stats.metrics
        ]
        names = (
            ["epoch"]
            + [f"train_{n}" for n in met_names]
            + [f"valid_{n}" for n in met_names]
            + ["time"]
        )
        self.logger(names)

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)

    def after_epoch(self):
        stats = [str(self.epoch)]
        for o in [self.train_stats, self.valid_stats]:
            stats += [f"{v:.6f}" for v in o.avg_stats]
        stats += [format_time(time.time() - self.start_time)]
        self.logger(stats)


class RecorderGAN(Callback):
    """
    Stores some values that will enable visualization and hooks to get model state
    """

    def begin_fit(self):
        self.lrsG, self.lrsD, self.losses_D, self.losses_G, self.nb_batches = (
            [0 for _ in self.opt_func_G.param_groups],
            [0 for _ in self.opt_func_D.param_groups],
            [],
            [],
            [],
        )

    def after_batch(self):
        if not self.in_train:
            return
        self.lrsG.append(self.opt_func_G.param_groups[-1]["lr"])
        self.lrsD.append(self.opt_func_D.param_groups[-1]["lr"])
        self.losses_G.append(self.loss_G.detach().cpu())
        self.losses_D.append(self.loss_D.detach().cpu())

    def after_epoch(self):
        self.nb_batches.append(self.epoch)

    def plot_lr(self):
        plt.plot(self.lrsG, label="gen lr")
        plt.plot(self.lrsD, label="disc lr ")
        plt.legend()

    def plot_loss(self):
        plt.plot(self.losses_D[: len(self.losses_D)], label="disc loss")
        plt.plot(self.losses_G[: len(self.losses_G)], label="gen loss")
        plt.legend()

    def plot(self):
        losses_D = [o.item() for o in self.losses_D]
        n = len(losses_D)
        plt.xscale("log")
        plt.plot(self.lrsD[:n], losses_D[:n], label="loss disc")
        plt.legend()

        losses_G = [o.item() for o in self.losses_G]
        n = len(losses_G)
        plt.xscale("log")
        plt.plot(self.lrsG[:n], losses_G[:n], label="loss gen")
        plt.legend()
