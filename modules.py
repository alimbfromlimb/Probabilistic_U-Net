import torch
from torch import nn
from dpipe.torch import TorchModel, set_lr, to_np, sequence_to_np, to_cuda
from dpipe.torch.utils import to_var, sequence_to_var


def kl_divergence(prior_out: torch.Tensor, post_out: torch.Tensor):
    mean_0, log_std_0 = prior_out[:, 0], prior_out[:, 1]
    mean_1, log_std_1 = post_out[:, 0], post_out[:, 1]

    std_delta = 2 * (log_std_1 - log_std_0)
    mean_delta = mean_1 - mean_0
    dim = mean_0.shape[1]

    trace = torch.exp(-std_delta)
    delta = torch.exp(-2 * log_std_1) * mean_delta ** 2
    total = trace.sum(1) + delta.sum(1) + std_delta.sum(1)

    res = ((total - dim) / 2).mean()

    return res

def compute_mmd(x, y):

    def compute_kernel(x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = to_cuda(x.unsqueeze(1))  # (x_size, 1, dim)
        y = to_cuda(y.unsqueeze(0)) # (1, y_size, dim)
        tiled_x = to_cuda(x.expand(x_size, y_size, dim))
        tiled_y = to_cuda(y.expand(x_size, y_size, dim))
        kernel_input = to_cuda((tiled_x - tiled_y).pow(2).mean(2) / float(dim))
        return to_cuda(torch.exp(-kernel_input))  # (x_size, y_size)

    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd


class AutoEncoder(nn.Module):
    def __init__(self, encoder_conv, encoder_lin, decoder_lin, decoder_conv):
        super().__init__()
        self.encoder_conv = encoder_conv
        self.encoder_lin = encoder_lin
        self.decoder_lin = decoder_lin
        self.decoder_conv = decoder_conv

    def forward(self, batch_of_contours):
        input_shape = batch_of_contours.shape
        batch_size = input_shape[0]
        x = nn.functional.interpolate(batch_of_contours, 256)
        x = self.encoder_conv(x)
        x = x.reshape(batch_size, -1)
        z = self.encoder_lin(x)
        x = self.decoder_lin(z)
        x = x.reshape(batch_size, 128, 32)
        x = self.decoder_conv(x)
        x = nn.functional.interpolate(x, input_shape[2:])
        return z, x


class Wnet(TorchModel):
    def __init__(self, logits2pred, logits2loss, dist_loss, optimize, cuda, u_net, autoencoder, to_curve, latent_space_dim,
                 beta=1, gama=1):
        self.u_net = u_net
        self.autoencoder = autoencoder
        self.dist_loss = dist_loss
        self.to_curve = to_curve
        self.latent_space_dim = latent_space_dim
        self.beta = beta
        self.gama = gama
        model_core = nn.ModuleDict({"u_net": self.u_net, "autoencoder": self.autoencoder})

        super().__init__(model_core, logits2pred, logits2loss, optimize, cuda)

    def do_train_step(self, batch_of_images, batch_of_contours, *, lr):
        """
        Performs a forward-backward pass, as well as the gradient step, according to the given ``inputs``.

        Notes
        -----
        Note that both input and output are **not** of type `torch.Tensor` - the conversion
        to torch.Tensor is made inside this function.
        """
        self.model_core.train()
        batch_of_images, batch_of_contours = sequence_to_var(batch_of_images, batch_of_contours, cuda=self.model_core)

        u_out = self.u_net(batch_of_images)
        u_curve = self.to_curve(u_out)

        z, reproduced_curve = self.autoencoder(u_curve)

        true_samples = Variable(torch.randn(200, self.latent_space_dim), requires_grad=False)

        loss_u_curve = self.logits2loss(u_out, batch_of_contours)
        loss_mmd = self.dist_loss(true_samples, z)
        loss_reproduced_curve = torch.nn.functional.mse_loss(u_curve, reproduced_curve)
        loss = loss_u_curve + self.beta*loss_mmd + self.gama*loss_reproduced_curve

        set_lr(self.optimizer, lr)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return sequence_to_np(loss, loss_u_curve, loss_mmd, loss_reproduced_curve)

    def do_inf_step(self, batch_of_images):
        """
        Returns the prediction for the given ``inputs``.

        Notes
        -----
        Note that both input and output are **not** of type `torch.Tensor` - the conversion
        to torch.Tensor is made inside this function.
        """
        self.model_core.eval()

        with torch.no_grad():
            batch_of_images = to_var(batch_of_images, self.model_core)
            u_out = self.u_net(batch_of_images)
            return to_np(self.logits2pred(u_out))

    def get_trio(self, batch_of_images):
        """
        Returns the prediction for the given ``inputs``.

        Notes
        -----
        Note that both input and output are **not** of type `torch.Tensor` - the conversion
        to torch.Tensor is made inside this function.
        """
        self.model_core.eval()

        with torch.no_grad():
            batch_of_images = to_var(batch_of_images, self.model_core)
            u_out = self.u_net(batch_of_images)
            u_curve = self.to_curve(u_out)
            z, reconstructed = self.autoencoder(u_curve)
            return to_np(self.logits2pred(u_out)), to_np(z), to_np(reconstructed)

    def z_to_curve(self, z, input):
        self.model_core.eval()

        with torch.no_grad():
            input_shape = input.shape
            batch_size = input_shape[0]
            z = to_var(z, self.model_core)
            x = self.autoencoder.decoder_lin(z)
            x = x.reshape(batch_size, 128, 32)
            x = self.autoencoder.decoder_conv(x)
            reproduced_curve = nn.functional.interpolate(x, input_shape[2:])
            return to_np(reproduced_curve)
        
        
class PosteriorNet(nn.Module):
    def __init__(self, x_net, y_net, mixer):
        super().__init__()
        self.x_net = x_net
        self.y_net = y_net
        self.mixer = mixer

    def forward(self, images, contours):
        batch_size = images.shape[0]
        x = self.x_net(images).reshape(batch_size, -1)
        y = self.y_net(contours).reshape(batch_size, -1)
        comb = torch.cat([x, y], 1)
        res = self.mixer(comb)
        # return to_cuda(torch.zeros_like(res))
        return res


class FinalNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def plant(self, batch_of_images: torch.Tensor, point: torch.Tensor):
        point_tensor = torch.ones(
            batch_of_images.shape[0], point.shape[1], *batch_of_images.shape[2:],
            device=batch_of_images.device, dtype=batch_of_images.dtype
        )

        for _ in batch_of_images.shape[2:]:
            point = point.unsqueeze(-1)

        res = torch.cat([batch_of_images, point_tensor * point], 1)
        return res

    def forward(self, images, point):
        huge_tensor = self.plant(images, point)
        res = self.net(huge_tensor)
        return res


class False_PriorNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, images):
        res = self.net(images)
        return to_cuda(torch.zeros_like(res))


def sample(distrib_tensor):
    means, log_stds = distrib_tensor[:, 0], distrib_tensor[:, 1]
    eps = torch.randn_like(means)
    res = means + eps * torch.exp(log_stds)
    return res


def visualize_sample(distrib_tensor: torch.Tensor, i, j):
    means, log_stds = distrib_tensor[:, 0], distrib_tensor[:, 1]
    stds = torch.exp(log_stds)
    x = i * stds[0][0] + means[0][0]
    y = j * stds[0][1] + means[0][1]
    l = [x, y]
    point = to_cuda(torch.Tensor(l))
    point = point[None, :]
    return point


class CondUnet(TorchModel):
    def __init__(self, logits2pred, logits2loss, dist_loss, optimize, cuda, u_net, prior_net, post_net, sample,
                 final_net):
        self.u_net = u_net
        self.prior_net = prior_net
        self.post_net = post_net
        self.sample = sample
        self.dist_loss = dist_loss
        self.final_net = final_net
        model_core = nn.ModuleDict({"u_net": self.u_net, "prior_net": self.prior_net, "post_net": self.post_net,
                                    "final_net": self.final_net})

        super().__init__(model_core, logits2pred, logits2loss, optimize, cuda)

    # functions for CondUnet

    def do_train_step(self, batch_of_images, batch_of_contours, *, lr, beta):
        """
        Performs a forward-backward pass, as well as the gradient step, according to the given ``inputs``.

        Notes
        -----
        Note that both input and output are **not** of type `torch.Tensor` - the conversion
        to torch.Tensor is made inside this function.
        """
        self.model_core.train()
        batch_of_images, batch_of_contours = sequence_to_var(batch_of_images, batch_of_contours, cuda=self.model_core)

        u_out = self.u_net(batch_of_images)
        prior_out = self.prior_net(batch_of_images)
        post_out = self.post_net(batch_of_images, batch_of_contours)
        point = self.sample(post_out)
        final_out = self.final_net(u_out, point)

        loss_u_net = self.logits2loss(final_out, batch_of_contours)
        loss_kl = self.dist_loss(prior_out, post_out)
        loss = loss_u_net + beta * loss_kl

        set_lr(self.optimizer, lr)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return sequence_to_np(loss, loss_u_net, loss_kl)

    def do_inf_step(self, batch_of_images):
        """
        Returns the prediction for the given ``inputs``.

        Notes
        -----
        Note that both input and output are **not** of type `torch.Tensor` - the conversion
        to torch.Tensor is made inside this function.
        """
        self.model_core.eval()

        with torch.no_grad():
            batch_of_images = to_var(batch_of_images, self.model_core)
            u_out = self.u_net(batch_of_images)
            prior_out = self.prior_net(batch_of_images)
            point = self.sample(prior_out)
            final_out = self.logits2pred(self.final_net(u_out, point))

            return to_np(final_out)
