import torch
import os
import re
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.bmp')


def _parse_frame_num(filename):
    m = re.match(r'(\d+)\.\w+$', filename)
    return int(m.group(1)) if m else 0


def collect_sequences(root_dir, min_frames=8):
    sequences = {}
    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=True):
        frames = sorted(
            [os.path.join(dirpath, f) for f in filenames
             if f.lower().endswith(IMG_EXTS) and 'Zone.Identifier' not in f],
            key=lambda p: _parse_frame_num(os.path.basename(p))
        )
        if len(frames) >= min_frames:
            sequences[dirpath] = frames
    return sequences


def collect_images(root_dir, exts=IMG_EXTS, recursive=True):
    image_paths = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=True):
            for fname in sorted(filenames):
                if fname.lower().endswith(exts) and 'Zone.Identifier' not in fname:
                    image_paths.append(os.path.join(dirpath, fname))
    else:
        for fname in sorted(os.listdir(root_dir)):
            if fname.lower().endswith(exts):
                image_paths.append(os.path.join(root_dir, fname))
    return image_paths


def collect_ycb_paired(root_dir):
    pairs = []
    data_files_dir = None
    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=True):
        if 'data_files' in dirnames:
            data_files_dir = os.path.join(dirpath, 'data_files')
            break
    if data_files_dir is None:
        data_files_dir = root_dir

    for obj_name in sorted(os.listdir(data_files_dir)):
        obj_path = os.path.join(data_files_dir, obj_name)
        if not os.path.isdir(obj_path):
            continue
        t_dir = os.path.join(obj_path, 'tactile_imgs')
        h_dir = os.path.join(obj_path, 'gt_height_map')
        if not (os.path.isdir(t_dir) and os.path.isdir(h_dir)):
            continue

        t_files = sorted([f for f in os.listdir(t_dir)
                          if f.lower().endswith(IMG_EXTS) and 'Zone.Identifier' not in f])
        h_files = sorted([f for f in os.listdir(h_dir)
                          if f.endswith('.npy') and 'Zone.Identifier' not in f])
        n = min(len(t_files), len(h_files))
        for i in range(n):
            pairs.append((
                os.path.join(t_dir, t_files[i]),
                os.path.join(h_dir, h_files[i]),
            ))
    return pairs


class FlatImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0


class PairedDataset(Dataset):
    def __init__(self, pairs, img_size=(128, 128)):
        self.pairs = pairs
        self.img_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        tactile_path, heightmap_path = self.pairs[idx]

        tactile_img = Image.open(tactile_path).convert('RGB')
        tactile_tensor = self.img_transform(tactile_img)

        heightmap = np.load(heightmap_path).astype(np.float32)
        h_min, h_max = heightmap.min(), heightmap.max()
        if h_max > h_min:
            heightmap = (heightmap - h_min) / (h_max - h_min)
        else:
            heightmap = np.zeros_like(heightmap)
        heightmap = torch.from_numpy(heightmap).unsqueeze(0)
        heightmap = torch.nn.functional.interpolate(
            heightmap.unsqueeze(0), size=self.img_transform.transforms[0].size,
            mode='bilinear', align_corners=False
        ).squeeze(0)

        return tactile_tensor, heightmap


class SequenceVolumeDataset(Dataset):
    def __init__(self, sequences, num_frames=4, transform=None):
        self.transform = transform
        self.num_frames = num_frames
        self.samples = []
        self._seq_dirs = []
        for seq_dir, frames in sequences.items():
            if len(frames) >= num_frames:
                self._seq_dirs.append(seq_dir)
                for start in range(0, len(frames) - num_frames + 1, max(1, num_frames // 2)):
                    self.samples.append((len(self._seq_dirs) - 1, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_idx, start = self.samples[idx]
        seq_dir = self._seq_dirs[seq_idx]

        seq_cache = getattr(self, '_seq_cache', None)
        if seq_cache is None:
            self._seq_cache = self._build_cache()
            seq_cache = self._seq_cache

        frames = seq_cache[seq_dir]
        imgs = []
        for i in range(self.num_frames):
            img = Image.open(frames[start + i]).convert('RGB')
            imgs.append(img)

        if self.transform:
            seed = torch.initial_seed()
            transformed = []
            for img in imgs:
                torch.manual_seed(seed)
                transformed.append(self.transform(img))
            imgs = transformed

        volume = torch.stack(imgs, dim=0)
        return volume, 0

    def _build_cache(self):
        cache = {}
        for seq_dir in self._seq_dirs:
            frames = sorted(
                [os.path.join(seq_dir, f) for f in os.listdir(seq_dir)
                 if f.lower().endswith(IMG_EXTS) and 'Zone.Identifier' not in f],
                key=lambda p: _parse_frame_num(os.path.basename(p))
            )
            cache[seq_dir] = frames
        return cache


def collect_ycb_by_object(root_dir):
    objects = {}
    data_files_dir = None
    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=True):
        if 'data_files' in dirnames:
            data_files_dir = os.path.join(dirpath, 'data_files')
            break
    if data_files_dir is None:
        for entry in sorted(os.listdir(root_dir)):
            obj_path = os.path.join(root_dir, entry)
            if not os.path.isdir(obj_path):
                continue
            t_dir = os.path.join(obj_path, 'tactile_imgs')
            if os.path.isdir(t_dir):
                imgs = sorted(
                    [os.path.join(t_dir, f) for f in os.listdir(t_dir)
                     if f.lower().endswith(IMG_EXTS) and 'Zone.Identifier' not in f],
                    key=lambda p: _parse_frame_num(os.path.basename(p))
                )
                if imgs:
                    objects[entry] = imgs

            h_dir = os.path.join(obj_path, 'gt_height_map')
            if os.path.isdir(t_dir) and os.path.isdir(h_dir):
                h_files = sorted(
                    [os.path.join(h_dir, f) for f in os.listdir(h_dir)
                     if f.endswith('.npy') and 'Zone.Identifier' not in f],
                    key=lambda p: _parse_frame_num(os.path.basename(p))
                )
                if h_files:
                    objects[entry + '_gt'] = h_files
        return objects

    for obj_name in sorted(os.listdir(data_files_dir)):
        obj_path = os.path.join(data_files_dir, obj_name)
        if not os.path.isdir(obj_path):
            continue
        t_dir = os.path.join(obj_path, 'tactile_imgs')
        if os.path.isdir(t_dir):
            imgs = sorted(
                [os.path.join(t_dir, f) for f in os.listdir(t_dir)
                 if f.lower().endswith(IMG_EXTS) and 'Zone.Identifier' not in f],
                key=lambda p: _parse_frame_num(os.path.basename(p))
            )
            if imgs:
                objects[obj_name] = imgs

        h_dir = os.path.join(obj_path, 'gt_height_map')
        if os.path.isdir(t_dir) and os.path.isdir(h_dir):
            h_files = sorted(
                [os.path.join(h_dir, f) for f in os.listdir(h_dir)
                 if f.endswith('.npy') and 'Zone.Identifier' not in f],
                key=lambda p: _parse_frame_num(os.path.basename(p))
            )
            if h_files:
                objects[obj_name + '_gt'] = h_files

    return objects


def data_loader_3d(args, root='./',
                   train_dir='dataset/train',
                   val_dir=None,
                   test_dir='dataset/test'):
    kwopt = {'num_workers': 8, 'pin_memory': True, 'prefetch_factor': 4, 'persistent_workers': True}
    w_size, h_size = int(16 * 8), int(16 * 8)
    num_frames = getattr(args, 'num_frames', 4)

    trn_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.RandomCrop(args.image_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.ToTensor(),
    ])

    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((w_size, h_size)),
        torchvision.transforms.ToTensor(),
    ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((w_size, h_size)),
        torchvision.transforms.ToTensor(),
    ])

    if val_dir is not None and os.path.isdir(os.path.join(root, val_dir)):
        train_seqs = collect_sequences(os.path.join(root, train_dir), min_frames=num_frames)
        val_seqs = collect_sequences(os.path.join(root, val_dir), min_frames=num_frames)
    else:
        train_seqs = collect_sequences(os.path.join(root, train_dir), min_frames=num_frames)
        val_seqs = {}

    # 限制每序列最大帧数 (防止超大训练集)
    max_frames = getattr(args, 'max_frames_per_seq', 0)
    if max_frames > 0:
        for k in list(train_seqs.keys()):
            if len(train_seqs[k]) > max_frames:
                train_seqs[k] = train_seqs[k][:max_frames]
        for k in list(val_seqs.keys()):
            if len(val_seqs[k]) > max_frames:
                val_seqs[k] = val_seqs[k][:max_frames]

    trn_dataset = SequenceVolumeDataset(train_seqs, num_frames=num_frames,
                                         transform=trn_transforms)
    trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True, **kwopt,
                            drop_last=False)

    val_dataset = SequenceVolumeDataset(val_seqs, num_frames=num_frames,
                                         transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwopt,
                            drop_last=False)

    test_imgs = collect_images(os.path.join(root, test_dir))
    test_dataset = FlatImageDataset(test_imgs, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **kwopt, drop_last=False)

    return trn_loader, val_loader, test_loader
