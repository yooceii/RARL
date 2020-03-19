from SimCLR.data_aug.data_transform import DataTransform, get_data_transform_opes
from SimCLR.utils import get_similarity_function
from SimCLR.utils import step as loss

import torch

import pdb

class SimCLR:
    def __init__(self, batch_size, device, temp=.5):
        self.batch_size = batch_size
        self.temperature = temp
        self.device = device

        #Data augmentation
        data_augment = get_data_transform_opes(s=1, crop_size=96)
        self.transform = DataTransform(data_augment)

        #Loss information
        _, simularity_func = get_similarity_function(True)
        self.sim = simularity_func
        self.n_mask = (1 - torch.eye(2*batch_size)).type(torch.bool).to(device)
        idx = torch.cat([torch.arange(batch_size-1, 2*batch_size-1), torch.arange(batch_size)])
        self.labels = torch.eye(2*batch_size, 2*batch_size-1)[idx].to(device)

        #Network
        self.softmax = torch.nn.Softamx(dim=-1)

    def update_encoder(self, obs):
        # SimCLR update
        aug_obs = obs.cpu().numpy()
        # Is = []
        # Js = []
        # for i in aug_obs:
        #     tmp_Is = []
        #     tmp_Js = []
        #     for c in i:
        #         a, b = transform.aug(c.astype(np.uint8))
        #         tmp_Is += [a]
        #         tmp_Js += [b]
        #     Is += [tmp_Is]
        #     Js += [tmp_Js]
        # Is = np.array(Is)
        # Js = np.array(Js)
        Is = np.random.rand(*obs.shape)
        Js = np.random.rand(*obs.shape)
        Is = torch.tensor(Is).float().cuda(cuda_id)
        Js = torch.tensor(Js).float().cuda(cuda_id)
        for _ in range(1):
            # Q = mp.Queue()
            # actor_critic.simclr.share_memory()
            # p1 = mp.Process(target=actor_critic.simclr, args=(Is,))
            # p2 = mp.Process(target=actor_critic.simclr, args=(Js,))
            # p1.start()
            # p2.start()
            # processes = [p1, p2]
            # for p in processes: p.join()
            # zis = actor_critic.simclr.get()
            # zjs = actor_critic.simclr.get()
            # assert actor_critic.simclr.Q.empty() is True

            zis = actor_critic.simclr(Is)
            zjs = actor_critic.simclr(Js)
            # print(zis.shape, zjs.shape)
            l = loss(zis, zjs, megative_mask, labels, similarity_func, batch_size, softmax, 0.5)
            l.backward()
            simclr_optimizer.step()
            # print("simclr update")

