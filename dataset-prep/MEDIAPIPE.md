# Mediapipe

**mediapipe version:** `0.10.21`

---

## Landmark values

**Common for Face, Pose, and Hands:**
- **X**: Normalized by image width → ideally in `[0.0, 1.0]` range (left to right).
- **Y**: Normalized by image height → ideally in `[0.0, 1.0]` range (top to bottom).
In practice, values can slightly exceed `[0.0, 1.0]` (such as, >1.0 or <0.0) when the model extrapolates positions for partially occluded or out-of-frame body parts.

**Z Coordinate (Depth):**
- Relative depth (not absolute distance).
- Smaller (more negative) Z = closer to the camera.
- The magnitude/scale of Z is roughly the same as X (_uses a similar unit to the normalized image width_).
- Range: Typically around `-1.0 to +1.0` or a bit wider, depending on the component, but no strict bounds.

**Pose Visibility:**

MediaPipe's Pose Landmarker (including when used in the Holistic pipeline), the model always outputs all 33 pose landmarks with their `X, Y, Z` coordinates, even if a body part is occluded, partially visible, or completely outside the image frame.
The visibility field (a value between `0.0 and 1.0`) indicates the model's estimated likelihood that the landmark is visible (not occluded by another body part or object) within the frame.

- A high visibility (such as, close to 1.0) means the model is confident the point is directly observable.
- A low visibility (such as, close to 0.0) means it's likely occluded or not visible, but the model still provides a predicted position based on context from the rest of the body (using its learned human pose priors).

---

## Landmark `.npy` File Format

Each recorded frame is saved as a NumPy `.npy` file containing a **flattened vector of Mediapipe landmarks**.

One file corresponds to **one video frame**.

### File Structure

- **Path**:  
  `{signer_id}/{sign}/landmarks/rep-{n}/frame-XX.npy`

- **Type**:  
  `numpy.ndarray`

- **Shape**:  
  `(1662,)`

- **Dtype**:  
  `float32` / `float64` (NumPy default)

- **Total `frame-*.npy` files**:  
  `82816`


## Vector Layout (Fixed Order)

The vector is a concatenation of four landmark groups, which are saved in a given order `return np.concatenate([face, pose, rh, lh])`:

### 1. Face Landmarks
- **468 points**
- **3 values per point**: `(x, y, z)`
- **Total values**: `468 × 3 = 1404`

### 2. Pose Landmarks
- **33 points**
- **4 values per point**: `(x, y, z, visibility)`
- **Total values**: `33 × 4 = 132`

> Even if some pose landmarks have `visibility = 0.0`, they are anyways saved into `.npy` and will be rendered in [`video-collector/dataset-checks/visualize_landmarks.py`](../video-collector/dataset-checks/04_visualize_landmarks.py).

### 3. Right Hand Landmarks
- **21 points**
- **3 values per point**: `(x, y, z)`
- **Total values**: `21 × 3 = 63`

### 4. Left Hand Landmarks
- **21 points**
- **3 values per point**: `(x, y, z)`
- **Total values**: `21 × 3 = 63`


## Total Vector Size

- 1404 (face)
- 132 (pose)
- 63 (right hand)
- 63 (left hand)
**= 1662 values**

<pre>
0                                                            1661
│──────────────────────────────────────────────────────────────│
│                                                              │
│  FACE (468×3)   POSE (33×4)   RIGHT HAND (21×3)  LEFT HAND   │
│  x y z ...      x y z v ...   x y z ...          x y z ...   │
│                                                              │
│  1404 values     132 values        63 values      63 values  │
│                                                              │
│───────────────┬─────────────┬────────────────┬───────────────│
0            1404          1536              1599          1662

v = visibility
</pre>


## Missing Landmarks

If a landmark group is **not detected** in a frame, its section is filled with **zeros** of the appropriate length with `np.zeros(A*B)`.
```python  
np.zeros(468*3) # face
np.zeros(33*4)  # pose
np.zeros(21*3)  # right hand
np.zeros(21*3)  # left hand
```

---

## Reconstructing Original Landmark Shapes

```python
import numpy as np

vec = np.load("frame-00.npy")

face = vec[0:1404].reshape(468, 3)
pose = vec[1404:1404+132].reshape(33, 4)
rh   = vec[1536:1536+63].reshape(21, 3)
lh   = vec[1599:1599+63].reshape(21, 3)
```

Alternatively (same result, no manual calculation needed):

```python
import numpy as np

vec = np.load("frame-00.npy")

face = vec[0:468*3].reshape((468, 3))
pose = vec[468*3:468*3 + 33*4].reshape((33, 4))
rh   = vec[468*3 + 33*4:468*3 + 33*4 + 21*3].reshape((21, 3))
lh   = vec[468*3 + 33*4 + 21*3:].reshape((21, 3))
```


