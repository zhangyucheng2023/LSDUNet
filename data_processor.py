import torch
import os
import re
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.bmp')


def _parse_frame_num(filename):
    m = re.match(r'(\d+)\.\w+$', filename)
    return int(m.group(1)) if m else 0


def collect_sequences(root_dir, min_frames=8, frame_filter=None):
    """Collect temporal sequences with minimum frame count."""
    sequences = {}
    for dirpath, _, filenames in os.walk(root_dir, followlinks=True):
        dir_name = os.path.basename(dirpath)
        frames = sorted(
            [os.path.join(dirpath, f) for f in filenames
             if f.lower().endswith(IMG_EXTS) and 'Zone.Identifier' not in f],
            key=lambda p: _parse_frame_num(os.path.basename(p))
        )
        if frame_filter is not None:
            frames = [p for p in frames
                      if (dir_name, os.path.basename(p)) in frame_filter]
        if len(frames) >= min_frames:
            sequences[dirpath] = frames
    return sequences


class SequenceVolumeDataset(Dataset):
    def __init__(self, sequences, num_frames=8, transform=None):
        self.transform = transform
        self.num_frames = num_frames
        self.samples = []
        self._seq_dirs = []
        for seq_dir, frames in sequences.items():
            if len(frames) >= num_frames:
                self._seq_dirs.append(seq_dir)
                for start in range(0, len(frames) - num_frames + 1, max(1, num_frames // 2)):
                    self.samples.append((len(self._seq_dirs) - 1, start))
        # 初始化时构建缓存，避免多进程 DataLoader 中每个 worker 重复构建
        self._seq_cache = self._build_cache()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_idx, start = self.samples[idx]
        seq_dir = self._seq_dirs[seq_idx]
        frames = self._seq_cache[seq_dir]
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
    """Collect YCB sequences organized by object."""
    objects = {}
    data_files_dir = None
    for dirpath, dirnames, _ in os.walk(root_dir, followlinks=True):
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
                    if os.path.isdir(h_dir):
                        h_files = sorted(
                            [os.path.join(h_dir, f) for f in os.listdir(h_dir)
                             if f.endswith('.npy') and 'Zone.Identifier' not in f],
                            key=lambda p: _parse_frame_num(os.path.basename(p))
                        )
                        if h_files:
                            objects[entry + '_gt'] = h_files
            else:
                imgs = sorted(
                    [os.path.join(obj_path, f) for f in os.listdir(obj_path)
                     if f.lower().endswith(IMG_EXTS) and 'Zone.Identifier' not in f],
                    key=lambda p: _parse_frame_num(os.path.basename(p))
                )
                if imgs:
                    objects[entry] = imgs
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


def load_tag_frame_split(split_path):
    """Load TAG frame-level split, returns set of (dir_name, filename)."""
    allowed = set()
    if not os.path.exists(split_path):
        return None
    with open(split_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            # 格式: "20220601_182052,0000033833.jpg,0,0"
            if len(parts) >= 2:
                seq_name = parts[0].strip()
                fname = parts[1].strip()
                allowed.add((seq_name, fname))
    return allowed


def collect_visgel_sequences(root_dir, max_recordings=None, min_frames=4):
    """Collect VisGel temporal sequences. Structure: touch/0/rec_00000/frame0000.jpg"""
    sequences = {}
    count = 0
    for split_dir in sorted(os.listdir(root_dir)):
        split_path = os.path.join(root_dir, split_dir)
        if not os.path.isdir(split_path):
            continue
        for rec_dir in sorted(os.listdir(split_path)):
            if max_recordings is not None and count >= max_recordings:
                break
            rec_path = os.path.join(split_path, rec_dir)
            if not os.path.isdir(rec_path):
                continue
            frames = sorted(
                [os.path.join(rec_path, f) for f in os.listdir(rec_path)
                 if f.lower().endswith(IMG_EXTS)],
                key=lambda p: int(''.join(filter(str.isdigit,
                                   os.path.splitext(os.path.basename(p))[0])) or 0)
            )
            if len(frames) >= min_frames:
                sequences[rec_path] = frames
                count += 1
        if max_recordings is not None and count >= max_recordings:
            break
    return sequences


def data_loader_3d(args, root='./',
                   train_dir='dataset/train',
                   val_dir=None,
                   train_split=None,
                   val_split=None):
    kwopt = {'num_workers': 4, 'pin_memory': True, 'prefetch_factor': 2}
    w_size, h_size = int(16 * 8), int(16 * 8)
    num_frames = getattr(args, 'num_frames', 8)

    trn_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((w_size, h_size)),
        torchvision.transforms.RandomCrop(args.image_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
    ])

    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((w_size, h_size)),
        torchvision.transforms.CenterCrop(args.image_size),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
    ])

    # 加载 split 过滤集
    train_filter = load_tag_frame_split(train_split) if train_split else None
    val_filter = load_tag_frame_split(val_split) if val_split else None

    if val_dir is not None and os.path.isdir(os.path.join(root, val_dir)):
        train_seqs = collect_sequences(os.path.join(root, train_dir),
                                       min_frames=num_frames,
                                       frame_filter=train_filter)
        val_seqs = collect_sequences(os.path.join(root, val_dir),
                                     min_frames=num_frames,
                                     frame_filter=val_filter)
    else:
        train_seqs = collect_sequences(os.path.join(root, train_dir),
                                       min_frames=num_frames,
                                       frame_filter=train_filter)
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

    return trn_loader, val_loader
