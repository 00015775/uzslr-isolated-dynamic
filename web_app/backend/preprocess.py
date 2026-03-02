import torch
import torch.nn as nn

FACE_LANDMARKS = 468
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21

MAX_LEN = 32
PAD_VALUE = -100.0
NUM_CLASSES = 50

NOSE = [1, 2, 98, 327]
LNOSE = [98]
RNOSE = [327]

LIP = [
    0,
    61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]

LLIP = [84, 181, 91, 146, 61, 185, 40, 39, 37, 87, 178, 88, 95, 78, 191, 80, 81, 82]
RLIP = [314, 405, 321, 375, 291, 409, 270, 269, 267, 317, 402, 318, 324, 308, 415, 310, 311, 312]

POSE = [500, 502, 504, 501, 503, 505, 512, 513]
LPOSE = [513, 505, 503, 501]
RPOSE = [512, 504, 502, 500]

REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]

RHAND = list(range(468 + 33, 468 + 33 + 21))
LHAND = list(range(468 + 33 + 21, 468 + 33 + 42))

POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE

NUM_NODES = len(POINT_LANDMARKS)
CHANNELS = 6 * NUM_NODES


def unpack_frame(vec):
    vec = torch.tensor(vec, dtype=torch.float32)
    face = vec[0:468 * 3].reshape(468, 3)
    pose = vec[468 * 3: 468 * 3 + 33 * 4].reshape(33, 4)
    pose = pose[:, :3]
    rh = vec[468 * 3 + 33 * 4: 468 * 3 + 33 * 4 + 21 * 3].reshape(21, 3)
    lh = vec[468 * 3 + 33 * 4 + 21 * 3:].reshape(21, 3)
    return torch.cat([face, pose, rh, lh], dim=0)


class Preprocess(nn.Module):
    def __init__(self, max_len=32, point_landmarks=POINT_LANDMARKS):
        super().__init__()
        self.max_len = max_len
        self.register_buffer(
            "landmark_idx",
            torch.tensor(point_landmarks, dtype=torch.long)
        )

    def forward(self, x):
        if x.dim() == 1 and x.shape[0] == 1662:
            frames = unpack_frame(x).unsqueeze(0)
        elif x.dim() == 2 and x.shape[1] == 1662:
            x = x.to(torch.float32)
            face = x[:, 0:468 * 3].reshape(-1, 468, 3)
            pose = x[:, 468 * 3: 468 * 3 + 33 * 4].reshape(-1, 33, 4)[:, :, :3]
            rh = x[:, 468 * 3 + 33 * 4: 468 * 3 + 33 * 4 + 21 * 3].reshape(-1, 21, 3)
            lh = x[:, 468 * 3 + 33 * 4 + 21 * 3:].reshape(-1, 21, 3)
            frames = torch.cat([face, pose, rh, lh], dim=1)
        elif x.dim() == 3 and x.shape[1:] == (543, 3):
            frames = x
        else:
            raise ValueError(f"Unexpected input shape {x.shape}")

        frames = frames[:, self.landmark_idx]
        frames = frames[..., :2]

        assert 17 in POINT_LANDMARKS, "Center landmark 17 missing"

        center = frames[:, self.landmark_idx.tolist().index(17):self.landmark_idx.tolist().index(17) + 1]
        center = torch.nanmean(center, dim=(0, 1), keepdim=True)
        center = torch.where(
            torch.isnan(center),
            torch.tensor(0.5, dtype=center.dtype, device=center.device),
            center
        )

        diff = frames - center
        diff = torch.where(torch.isnan(diff), torch.zeros_like(diff), diff)
        std = torch.std(diff, dim=(0, 1), keepdim=True, unbiased=False)
        std = torch.clamp(std, min=1e-6)
        frames = diff / std

        dx = torch.zeros_like(frames)
        dx[1:] = frames[1:] - frames[:-1]

        dx2 = torch.zeros_like(frames)
        dx2[:-2] = frames[2:] - frames[:-2]

        frames = frames.reshape(frames.shape[0], -1)
        dx = dx.reshape(dx.shape[0], -1)
        dx2 = dx2.reshape(dx2.shape[0], -1)

        return torch.cat([frames, dx, dx2], dim=-1)
