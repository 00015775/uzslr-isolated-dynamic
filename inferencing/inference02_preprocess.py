import torch
import torch.nn as nn

# simply copied the preprocessing code experimented on 03_ak_model_dev_v1.ipynb

FACE_LANDMARKS = 468
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21

MAX_LEN = 32              # fixed
PAD_VALUE = -100.0        # shorter frames are not padded
NUM_CLASSES = 50          # 50 uzbek signs

NOSE=[
    1,2,98,327
]
# additional info
LNOSE = [98]
RNOSE = [327]

LIP = [ 0, 
    61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]

# additional info
LLIP = [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82]
RLIP = [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312]

# as of now testing without a pose
POSE = [500, 502, 504, 501, 503, 505, 512, 513]
LPOSE = [513,505,503,501]
RPOSE = [512,504,502,500]

REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]
LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
]

RHAND = list(range(468 + 33, 468 + 33 + 21))      # [501, 502, ..., 522] 
LHAND = list(range(468 + 33 + 21, 468 + 33 + 42)) # [522, 523, ..., 542]


POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE 

NUM_NODES = len(POINT_LANDMARKS)  # 118 selected features
CHANNELS = 6 * NUM_NODES          # 708 total output features

# 6 is chosen because (2 channels -> (x, y) × 3 -> position, first difference:velocity, second difference:acceleration) = (2*3) = 6
# out of (x,y,z) and visibility, only (x,y) values are chosen as feature inputs
print(NUM_NODES, CHANNELS)


# Convert (1662,) numpy ndarray -> (543,3) torch tensor
def unpack_frame(vec):
    """
    vec: (1662,)
    returns: (543, 3)
    """
    vec = torch.tensor(vec, dtype=torch.float32)
    
    face = vec[0:468*3].reshape(468, 3)

    pose = vec[468*3 : 468*3 + 33*4].reshape(33, 4)
    pose = pose[:, :3]  # drop visibility

    rh = vec[468*3 + 33*4 : 468*3 + 33*4 + 21*3].reshape(21, 3)
    lh = vec[468*3 + 33*4 + 21*3 :].reshape(21, 3)

    return torch.cat([face, pose, rh, lh], dim=0) # (543, 3)


class Preprocess(nn.Module):
    def __init__(self, max_len=32, point_landmarks=POINT_LANDMARKS):
        super().__init__()
        self.max_len = max_len
        self.register_buffer(
            "landmark_idx",
            torch.tensor(point_landmarks, dtype=torch.long)
        )

    def forward(self, x):
        """
        x: (1662,) single frame or, 
           (T, 1662) stacked frames or,
           (T, 543, 3) only if augmentation is applied, which unpacks 
        returns: (T, 6 * NUM_NODES)
        """
        
        """
        # as a sanity check, but not considered right now
        if isinstance(vec, np.ndarray):
            vec = torch.from_numpy(vec).to(torch.float32)
        else:
            vec = vec.to(torch.float32)
        """
            
        # automatically unpack if needed
        if x.dim() == 1 and x.shape[0] == 1662:
            # single frame (1662,)
            frames = unpack_frame(x).unsqueeze(0)  # (1, 543, 3)
        elif x.dim() == 2 and x.shape[1] == 1662:
            # stacked frames (T, 1662)
            # vectorized unpacking, should be much faster than calling unpack_frame 32 times for each repetition
            x = x.to(torch.float32)
            
            face = x[:, 0:468*3].reshape(-1, 468, 3)
            pose = x[:, 468*3 : 468*3 + 33*4].reshape(-1, 33, 4)[:, :, :3] # drop visibility
            rh = x[:, 468*3 + 33*4 : 468*3 + 33*4 + 21*3].reshape(-1, 21, 3)
            lh = x[:, 468*3 + 33*4 + 21*3 :].reshape(-1, 21, 3)
            
            frames = torch.cat([face, pose, rh, lh], dim=1)  # (T, 543, 3)
        elif x.dim() == 3 and x.shape[1:] == (543, 3):
            # already unpacked (T, 543, 3)
            frames = x
        else:
            raise ValueError(f"Unexpected input shape {x.shape}")

        
        # gather only selected 118 landmarks
        frames = frames[:, self.landmark_idx]  # (T, N, 3)

        # use only x,y columns
        frames = frames[..., :2]  # (T, N, 2)

        assert 17 in POINT_LANDMARKS, "Center landmark 17 missing" # helps to spot missing 17th landmark error
        
        #assert 0 in POINT_LANDMARKS, "Center landmark 0 (nose tip) missing"

        """
         center using landmark 17 (nose reference)
         here the center normalization is global over time (one single center for the entire sequence), as per code of Sohn. H (2023)
         however, experimenting with center normalization per-frame and not over entire sequence could be tried out
         if per-frame approach is tried out, then comment out these below 3 lines of center related code

         it turns out that 17th landmark is not the nose but rather is lip, but 0th landmark is the tip of the nose
         however after analysis, it was found that 17th landmark is more stable 
         since it is usually located close to the center ([0.5, 0.5])
        """

        #center_idx = self.landmark_idx.tolist().index(0)
        #center = frames[:, center_idx:center_idx+1]
        
        center = frames[:, self.landmark_idx.tolist().index(17):self.landmark_idx.tolist().index(17)+1]
        
        center = torch.nanmean(center, dim=(0,1), keepdim=True)
        center = torch.where(torch.isnan(center), torch.tensor(0.5, dtype=center.dtype, device=center.device), center)
        
        """
        # comment above 3 lines of "center" code, and uncomment the below 4 lines "center" code if it is being exprimented
        # may be experimented in the future
        # per-frame center normalization
        center_idx = self.landmark_idx.tolist().index(17)
        center = frames[:, center_idx:center_idx+1]       # (T,1,2)
        center = torch.nanmean(center, dim=1, keepdim=True)  # mean per frame
        center = torch.nan_to_num(center, nan=0.5)
        """
        
        # normalize relative to center
        diff = frames - center
        diff = torch.where(torch.isnan(diff), torch.zeros_like(diff), diff)  # replace NaN with 0
        std = torch.std(diff, dim=(0, 1), keepdim=True, unbiased=False)  #fixed standard deviation
        std = torch.clamp(std, min=1e-6)
        
        frames = diff / std

        # capturing temporal dynamics
        # velocity
        dx = torch.zeros_like(frames)
        dx[1:] = frames[1:] - frames[:-1]

        # acceleration
        dx2 = torch.zeros_like(frames)
        dx2[:-2] = frames[2:] - frames[:-2] # 2nd order difference
        """
        dx2[2:] = dx[2:] - dx[1:-1] also provides acceleration, but the value is dependent on first derivate
        if any changes happen to dx, so is dx2 effected by it.

        initially, it was not even acceleration, this code used to be: dx2[2:] = frames[2:] - frames[:-2] 
        which is 2nd-order/step difference and NOT the acceleration
        """

        # flatten per frame
        frames = frames.reshape(frames.shape[0], -1) 
        dx     = dx.reshape(dx.shape[0], -1)
        dx2    = dx2.reshape(dx2.shape[0], -1)

        return torch.cat([frames, dx, dx2], dim=-1) # (T, 708)
    
  