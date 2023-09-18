import pandas as pd
import numpy as np
import smartbody_skeleton_imu
import os
import matplotlib.pyplot as plt


def RodriguesMatrixModel(src, dst):
    # estimate the paras of converting two system (src,dst)
    scale = np.sum(np.sqrt(np.sum((dst - np.mean(dst, axis=0)) ** 2, axis=1))) / np.sum(
        np.sqrt(np.sum((src - np.mean(src, axis=0)) ** 2, axis=1)))
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src = src - src_mean
    dst = dst - dst_mean
    H = np.dot(src.T, dst)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = dst_mean.T - scale * np.dot(R, src_mean.T)
    return R, t, scale


def KptOut(R, t, scale, kpt_s):
    # convert kpt_s with different system
    kpt_all = []
    for i in range(kpt_s.shape[0]):
        kpt_new = []
        for kpt in kpt_s[i]:
            x_d, y_d, z_d = kpt[0], kpt[1], kpt[2]
            LN = np.row_stack((x_d, y_d, z_d))
            t = np.row_stack(t)
            S = np.dot(scale, R)
            N_c = np.dot(S, LN) + t
            kpt_new.append(N_c)
        kpt_all.append(kpt_new)
    kpt_all = np.array(kpt_all).reshape(len(kpt_all), -1, 3)
    return kpt_all


def distance(sub_cv, sub_imu):
    dists = []
    for s_cv, s_imu in zip(sub_cv, sub_imu):
        dist = np.linalg.norm(s_cv - s_imu, axis=0).mean()
        dists.append(dist)
    return sum(dists) / len(dists)


def synchronize(cv_angle, imu_angle):
    # use brute-force to synchronize the time
    z = [[0] * 50 for _ in range(1, 21)]
    i, j = 0, 0
    min_dist = 100
    for k in range(1, 21):
        j = 0
        for s in range(-10, 40):
            sub_cv = np.array(cv_angle[100:300]).reshape(-1, 17, 3)[:, joints]
            sub_imu = np.array(imu_angle[::k][100 + s:300 + s]).reshape(-1, 17, 3)[:, joints]
            dist = distance(sub_cv, sub_imu)
            z[i][j] = dist
            if dist < min_dist:
                min_dist = dist
                k_ = k
                s_ = s
            j += 1
        i += 1
    print("k: ", k_)
    print("s: ", s_)
    z = list(np.array(z).T)
    odf1 = np.linspace(1, 20, 20)
    odf2 = np.linspace(-10, 39, 50)
    X, Y = np.meshgrid(odf1, odf2)
    plt.title("distance", fontsize=20, fontname="Times New Roman")
    C = plt.contour(X, Y, z, 8, colors='black')
    plt.contourf(X, Y, z, 8, cmap=plt.cm.hot)
    plt.clabel(C, inline=1, fontsize=10)
    plt.colorbar()
    plt.show()


def write_smartbody_bvh(prediction3dpoint, name):
    # transform kpts to bvh
    point_changes = [6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10]
    # point_change = [6, 3, 4, 5, 2, 1, 0, 7, 8, 16, 9, 12, 11, 10, 13, 14, 15]

    for i in range(len(prediction3dpoint)):
        prediction3dpoint[i] = prediction3dpoint[i][point_changes, :]

    bvhfileDirectory = os.path.join("imu", "bvh")
    if not os.path.exists(bvhfileDirectory):
        os.makedirs(bvhfileDirectory)
    bvhfileName = os.path.join("imu", "bvh", "{}.bvh".format(name))

    SmartBody_skeleton = smartbody_skeleton_imu.SmartBodySkeleton()
    channels, header = SmartBody_skeleton.poses2bvh(prediction3dpoint, output_file=bvhfileName)

    return channels

# imu系统的关键点顺序
imu_name = ['Hips', 'LeftHip', 'LeftKnee', 'LeftAnkle', 'RightHip', 'RightKnee', 'RightAnkle', 'Chest2',
            'LeftShoulder', 'LeftElbow', 'LeftWrist', 'RightShoulder', 'RightElbow', 'RightWrist', 'Neck', 'Head',
            'HeadEnd']

# cv系统的关键点顺序
kpts_name = ['RightAnkle', 'RightKnee', 'RightHip', 'LeftHip', 'LeftKnee', 'LeftAnkle', 'Hips', 'Chest', 'Neck',
             'Head', 'RightWrist', 'RightElbow', 'RightShoulder', 'LeftShoulder', 'LeftElbow', 'LeftWrist', 'Chin']

# point_change = [6, 5, 4, 1, 2, 3, 0, 7, 8, 16, 14, 13, 12, 9, 10, 11, 15]

# change the order of kpts
point_change = [6, 5, 4, 1, 2, 3, 0, 7, 14, 16, 13, 12, 11, 8, 9, 10, 15]

# choose the valid kpts
# human36m_joints = [10, 11, 15, 14, 1, 4, 2, 3]
human36m_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]
# joints = [0, 2, 5, 12, 13, 15, 16]
joints = [0, 2, 5]

# download the kpts
kpt_s = np.load('imu_kpt.npy')
kpt_imu = kpt_s[0]
prediction = np.load('cv_kpt.npy')
kpt_cv = prediction[0]

if __name__ == '__main__':
    R, t, scale = RodriguesMatrixModel(kpt_imu[human36m_joints], kpt_cv[human36m_joints])

    # 转化关键点的坐标到同一坐标系
    kpt_all = KptOut(R, t, scale, kpt_s)

    # cv_angle = write_smartbody_bvh(prediction, 'cv0')
    # imu_angle = write_smartbody_bvh(kpt_all, 'imu0')
    # synchronize(cv_angle, imu_angle)
    np.save('./imu_0.npy', kpt_all, allow_pickle=True)
