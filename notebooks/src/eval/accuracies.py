# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
# Taken from https://github.com/Microsoft/human-pose-estimation.pytorch
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np



def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm, thr)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def keypoint_3d_pck(pred, gt, mask, stds, means, alignment='none', threshold=5.0):
    """Calculate the Percentage of Correct Keypoints (3DPCK) w. or w/o rigid
    alignment.
    Paper ref: `Monocular 3D Human Pose Estimation In The Wild Using Improved
    CNN Supervision' 3DV`2017
    More details can be found in the `paper
    <https://arxiv.org/pdf/1611.09813>`__.
    batch_size: N
    num_keypoints: K
    keypoint_dims: C
    Args:
        pred (np.ndarray[N, K, C]): Predicted keypoint location.
        gt (np.ndarray[N, K, C]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        alignment (str, optional): method to align the prediction with the
            groundtruth. Supported options are:
            - ``'none'``: no alignment will be applied
            - ``'scale'``: align in the least-square sense in scale
            - ``'procrustes'``: align in the least-square sense in scale,
                rotation and translation.
        threshold:  If L2 distance between the prediction and the groundtruth
            is less then threshold, the predicted result is considered as
            correct. Default: 0.15 (pixels).
    Returns:
        pck: percentage of correct keypoints.
    """
    assert mask.any()
    mask = mask.astype(np.bool)
    
    pred_norm = (pred*np.stack([stds]*len(pred)))+np.stack([means]*len(pred))
    gt_norm = (gt*np.stack([stds]*len(pred)))+np.stack([means]*len(pred))


    if alignment == 'none':
        pass
    elif alignment == 'procrustes':
        pred_norm = np.stack([
            compute_similarity_transform(pred_i, gt_i)
            for pred_i, gt_i in zip(pred_norm, gt_norm)
        ])
    elif alignment == 'scale':
        pred_dot_pred = np.einsum('nkc,nkc->n', pred_norm, pred_norm)
        pred_dot_gt = np.einsum('nkc,nkc->n', pred_norm, gt_norm)
        scale_factor = pred_dot_gt / pred_dot_pred
        pred_norm = pred_norm * scale_factor[:, None, None]
    else:
        raise ValueError(f'Invalid value for alignment: {alignment}')
    #  For depth function axis=-1 else ord = 2
    error = np.linalg.norm(pred_norm - gt_norm, ord=2, axis=-1)
    # error = np.absolute(pred_norm-gt_norm)
    pck = (error < threshold).astype(np.float32)[mask].mean()

    return pck



def keypoint_depth_pck(pred, gt, mask, stds, means, alignment='none', threshold=5.0):
    """Calculate the Percentage of Correct Keypoints (PCK)
    Returns:
        pck: percentage of correct keypoints.
    """
    assert mask.any()
    mask = mask.astype(np.bool)
    
    pred_norm = (pred*np.stack([stds]*len(pred)))+np.stack([means]*len(pred))
    gt_norm = (gt*np.stack([stds]*len(pred)))+np.stack([means]*len(pred))

    if alignment == 'none':
        pass
    else:
        raise ValueError(f'Invalid value for alignment: {alignment}')
    #  For depth function axis=-1 else ord = 2
    error = np.absolute(pred_norm-gt_norm)
    pck = (error < threshold).astype(np.float32)[mask].mean()

    return pck


# ------------------------------------------------------------------------------
# Adapted from https://github.com/akanazawa/hmr
# Original licence: Copyright (c) 2018 akanazawa, under the MIT License.
# ------------------------------------------------------------------------------
def compute_similarity_transform(source_points, target_points):
    """Computes a similarity transform (sR, t) that takes a set of 3D points
    source_points (N x 3) closest to a set of 3D points target_points, where R
    is an 3x3 rotation matrix, t 3x1 translation, s scale. And return the
    transformed 3D points source_points_hat (N x 3). i.e. solves the orthogonal
    Procrutes problem.
    Notes:
        Points number: N
    Args:
        source_points (np.ndarray([N, 3])): Source point set.
        target_points (np.ndarray([N, 3])): Target point set.
    Returns:
        source_points_hat (np.ndarray([N, 3])): Transformed source point set.
    """

    assert target_points.shape[0] == source_points.shape[0]
    assert target_points.shape[1] == 3 and source_points.shape[1] == 3

    source_points = source_points.T
    target_points = target_points.T

    # 1. Remove mean.
    mu1 = source_points.mean(axis=1, keepdims=True)
    mu2 = target_points.mean(axis=1, keepdims=True)
    X1 = source_points - mu1
    X2 = target_points - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, _, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Transform the source points:
    source_points_hat = scale * R.dot(source_points) + t

    source_points_hat = source_points_hat.T

    return source_points_hat