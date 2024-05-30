from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class VideoDataset(Dataset):

    def __init__(
        self,
        path,
        transform=None,
        frame_size=4,
    ):
        if transform is None:
            raise RuntimeError("Transform must be applied")

        self.frame_size = frame_size
        self.transform = transform
        self.image_paths = sorted(Path(path).iterdir())

    def __getitem__(self, index):
        start_idx = index * self.frame_size
        end_idx = (index + 1) * self.frame_size
        frame_paths = self.image_paths[start_idx:end_idx]

        frames = [
            self.transform(Image.open(p).convert("RGB")) for p in frame_paths
        ]

        return frames

    def __len__(self):
        return len(self.image_paths) // self.frame_size

