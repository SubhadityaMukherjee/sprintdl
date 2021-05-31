import matplotlib.animation as animation
import torch
import torchvision.utils as vutils
from IPython.display import HTML

from ..aug import pil_from_tensor
from ..core import *

Tensor = torch.cuda.FloatTensor


class GANLearner(Learner):
    def __init__(
        self,
        model_D,
        model_G,
        data,
        loss_func_D,
        loss_func_G,
        opt_func_D=sgd_opt,
        opt_func_G=sgd_opt,
        lr=1e-2,
        splitter=param_getter,
        cbs=None,
        cb_funcs=None,
        device="cuda:0",
        latent_dim=100,
        clip_value=0.01,
        n_critic=5,
    ):
        (
            self.model_D,
            self.model_G,
            self.data,
            self.loss_func_D,
            self.loss_func_G,
            self.opt_func_D,
            self.opt_func_G,
            self.lr,
            self.splitter,
        ) = (
            model_D,
            model_G,
            data,
            loss_func_D,
            loss_func_G,
            opt_func_D,
            opt_func_G,
            lr,
            splitter,
        )
        self.latent_dim = latent_dim
        self.c = len(set(self.data.train_ds.y.items))
        self.clip_value = clip_value
        self.in_train, self.logger, self.opt = False, print, None
        self.n_critic = n_critic
        self.img_list = []
        self.device = device
        self.fixed_noise = torch.randn(
            self.data.train_ds.x[0].size(1), self.latent_dim, device=self.device
        )
        #             print(self.fixed_noise.size())
        self.cbs = []
        self.add_cb(TrainEvalCallback())
        self.add_cbs(cbs)
        self.add_cbs(cbf() for cbf in listify(cb_funcs))

    def one_batch(self, i, xb, yb):
        try:
            self.iter = i
            self.xb, self.yb = xb, yb
            self.xb, self.yb = self.xb.to(self.device), self.yb.to(self.device)

            self("begin_batch")

            for _ in range(self.n_critic):
                noise = torch.randn(
                    self.data.train_dl.batch_size, self.latent_dim, 1, 1
                ).to(self.device)
                fake = self.model_G(noise)
                with torch.enable_grad():
                    disc_real = self.model_D(self.xb).reshape(-1)
                    disc_fake = self.model_D(fake).reshape(-1)
                    self.loss_D = self.loss_func_D(disc_real, disc_fake)
                self.pred = disc_fake
                self.model_D.zero_grad()
                self.loss_D.backward(retain_graph=True)
                self.opt_func_D.step()

                for p in self.model_D.parameters():
                    p.data.clamp_(-self.clip_value, self.clip_value)

            with torch.enable_grad():
                gen_fake = self.model_D(fake).reshape(-1)
            self("after_pred")
            with torch.enable_grad():
                self.loss_G = self.loss_func_G(gen_fake)

            self("after_loss")
            self.model_G.zero_grad()
            self("begin_backward")
            self.loss_G.backward()
            self("after_backward")
            self.opt_func_G.step()
            self.model_G.eval()
            self.model_D.eval()

            with torch.no_grad():
                self.img_list.append(vutils.make_grid(fake[:32], normalize=True))
            self.model_G.train()
            self.model_D.train()

            if not self.in_train:
                return
            self("after_step")
        except CancelBatchException:
            self("after_cancel_batch")
        finally:
            self("after_batch")

    def plot_output(self):
        return pil_from_tensor(self.img_list[-1])

    def animate_gan_output(self):
        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        ims = [
            [plt.imshow(np.transpose(i.cpu(), (1, 2, 0)), animated=True)]
            for i in self.img_list
        ]
        ani = animation.ArtistAnimation(
            fig, ims, interval=1000, repeat_delay=1000, blit=True
        )

        return HTML(ani.to_jshtml())

    def do_begin_fit(self, epochs):
        self.epochs, self.loss_G, self.loss_D = (
            epochs,
            Variable(tensor(0.0), requires_grad=True),
            Variable(tensor(0.0), requires_grad=True),
        )
        self("begin_fit")

    def fit(self, epochs, cbs=None, reset_opt=False):
        self.add_cbs(cbs)

        try:
            self.do_begin_fit(epochs)
            for epoch in range(epochs):
                if not self.do_begin_epoch(epoch):
                    self.all_batches()

                with torch.no_grad():
                    self.dl = self.data.valid_dl
                    if not self("begin_validate"):
                        self.all_batches()
                self("after_epoch")

        except CancelTrainException:
            self("after_cancel_train")
        finally:
            self("after_fit")
            self.remove_cbs(cbs)
