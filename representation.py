import os

from SimCLR.models.resnet_simclr import ResNetSimCLR
from SimCLR.data_aug.guassian_blur import GuassianBlur
from SimCLR.loss.nt_xent import NTXentLoss

from arguments import get_args

import torch
import torchvision.transforms as transforms

import pdb

class SimCLR:
    def __init__(self, input_shape, writer, device):
        args = get_args()
        self.num_steps = args.num_steps
        self.num_processes = args.num_processes
        self.temperature = args.contrastive_loss_temp
        self.writer = writer
        self.device = device
        self.model_checkpoints = os.path.join(writer.log_dir, 'encoder_checkpoints')
        self.save_interval = args.save_interval
        self.updates = 0

        self.data = torch.zeros(num_steps, num_processes, *input_shape).to(device)
        self.step = 0

        #Data augmentation
        s = args.color_jitter_magnitude
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.input_shape[0]),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
                                              transforms.ToTensor()])
        #Networks
        self.model = ResNetSimCLR(args.encoding_size, args.encoder_model)
        self._load_pretrained_weights(args.saved_encoder)
        self.model.to(device)

        #Optimization
        use_cosine = not args.use_dot_similarity
        self.criterion = NTXentLoss(device, self.num_steps*self.num_processes, self.temperature, use_cosine)

        self.optimizer = torch.optim.Adam(self.model.parameters(), 3e-4, weight_decay=args.simclr_weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=0, last_epoch=-1)

    def insert(self, obs):
        self.data[self.step].copy_(obs)
        self.step = (self.step + 1) % self.num_steps

    def encode(self, obs):
        with torch.no_grad():
            encoding, _ = self.model(obs)
        return encoding

    def update_encoder(self):
        for update in range(1):
            self.optimizer.zero_grad()

            xis, xjs = self._transform(self.data)
            xis = xis.to(self.device)
            xjs = xjs.to(self.device)

            loss = self._step(xis, xjs)
            loss.backward()

            self.writer.add_scalar('Representation/train_loss', loss.item(), global_step=self.updates)

            self.optimizer.step()
            self.updates += 1

            #No validation test yet (besides RL performance)
            # validate the model if requested
            #if epoch_counter % self.config['eval_every_n_epochs'] == 0:
            #    valid_loss = self._validate(model, valid_loader)
            #    if valid_loss < best_valid_loss:
            #        # save the model weights
            #        best_valid_loss = valid_loss
            #        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

            #    self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
            #    valid_n_iter += 1

            if(self.updates % self.save_interval == 0):
                torch.save(self.model.save_dict(), os.path.join(self.model_checkpoints, 'model.pth'))

            # warmup for the first 10 epochs
            if(self.updates/self.save_intervale >= 10):
                self.scheduler.step()
            self.writer.add_scalar('Representation/cosine_lr_decay', self.scheduler.get_lr()[0], global_step=self.updates)

    def replay_buffer(self):
        raise Exception('Not implemented, could store buffer to mix in update_encoder for more stable updates')

    def _step(self, xis, xjs):
        # get the representations and the projections
        ris, zis = self.model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = self.model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.criterion(zis, zjs)
        return loss

    def _transform(self, obs):
        s, p, d, h, w = obs.size()
        obs = obs.view(-1, d, h, w)
        xis = self.transforms(obs)
        xjs = self.transforms(obs)
        return xis, xjs

    def _load_pretrained_weights(self, file):
        if(file is not None):
            try:
                checkpoints_folder = os.path.join('./runs', file, 'encoder_checkpoints')
                state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
                model.load_state_dict(state_dict)
                print("Loaded pre-trained model with success.")
            except FileNotFoundError:
                print("Pre-trained weights not found. Training from scratch.")
