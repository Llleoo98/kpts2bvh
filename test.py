import smartbody_skeleton_imu
import os
import numpy as np


def write_smartbody_bvh(prediction3dpoint, name):
    # 将三维关键点转化为bvh文件
    # ！！！需要确保关键点的顺序与bvh的层级顺序一致
    point_changes = [6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10]

    for i in range(len(prediction3dpoint)):
        prediction3dpoint[i] = prediction3dpoint[i][point_changes, :]

    bvhfileDirectory = os.path.join("bvh")
    if not os.path.exists(bvhfileDirectory):
        os.makedirs(bvhfileDirectory)
    bvhfileName = os.path.join("bvh", "{}.bvh".format(name))

    SmartBody_skeleton = smartbody_skeleton_imu.SmartBodySkeleton()
    channels, header = SmartBody_skeleton.poses2bvh(prediction3dpoint, output_file=bvhfileName)

    return channels

if __name__ == '__main__':
    path = 'keypoints.npy' # 三维坐标的位置信息
    kpts = np.load(path)
    write_smartbody_bvh(kpts, 'kpts')
