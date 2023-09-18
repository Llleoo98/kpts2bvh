# kpts2bvh

This code aims to convert 3d keypoints to BVH files and synchronize two different coordinate systems.

The order of joint points are: 

`['RightAnkle', 'RightKnee', 'RightHip', 'LeftHip', 'LeftKnee', 'LeftAnkle', 'Hips', 'Chest', 'Neck', 'Head', 'RightWrist', 'RightElbow', 'RightShoulder', 'LeftShoulder', 'LeftElbow', 'LeftWrist', 'Chin']`

Run
```python test.py``` to convert the 3d keypoints to BVH files.

Run
```python synchronize.py``` to synchronize two different coordinate systems.
