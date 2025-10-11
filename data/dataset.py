import random
import torch.utils.data as data
import utils.utils_image as util
import torchvision.transforms.functional as F


class Dataset(data.Dataset):

    def __init__(self, opt):
        super(Dataset, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        if 'patch_size' in opt and opt['patch_size']:
            self.patch_size = opt['patch_size']
        else:
            self.patch_size = 64

        # -------------------------------------
        # get the path of H, return None if input is None
        # -------------------------------------
        if len(opt['data_root']) > 1:
            self.paths_H = []
            for root in opt['data_root']:
                self.paths_H.extend(util.get_image_paths(root))
        else:
            self.paths_H = util.get_image_paths(opt['data_root'][0])

    def __getitem__(self, index):
        # -------------------------------------
        # get H image
        # -------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        H, W = img_H.shape[:2]

        # ---------------------------------
        # randomly crop the patch
        # ---------------------------------
        self.patch_size_plus8 = self.patch_size+8
        # ---------------------------------
        # randomly crop the patch
        # ---------------------------------
        rnd_h = random.randint(0, max(0, H - self.patch_size_plus8))
        rnd_w = random.randint(0, max(0, W - self.patch_size_plus8))
        patch_H = img_H[rnd_h:rnd_h + self.patch_size_plus8, rnd_w:rnd_w + self.patch_size_plus8, :]

        # ---------------------------------
        # augmentation - flip, rotate
        # ---------------------------------
        mode = random.randint(0, 7)
        patch_H = util.augment_img(patch_H, mode=mode)

        img_H = patch_H.copy()

        H, W = img_H.shape[:2]
        if random.random() > 0.5:
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
        else:
            rnd_h = 0
            rnd_w = 0
        img_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]

        img_H = util.uint2tensor3(img_H)

        # follow the setting in osediff
        # target images scaled to -1, 1
        img_H = F.normalize(img_H, mean=[0.5], std=[0.5])

        return {'H': img_H}

    def __len__(self):
        return len(self.paths_H)
