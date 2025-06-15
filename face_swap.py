import numpy as np
import cv2
from PIL import Image

from utils import draw_image_landmarks, draw_image_landmarks_name, get_landmarks_points106, images_to_grid, \
    get_landmarks_points68, get_face_direction
from skimage import transform as trans
from color_transfer import color_transfer


def extract_face(img, landmarks):
    p = [1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0, 24, 23, 22, 21, 20, 19, 18, 32, 31, 30, 29, 28, 27,
         26, 25, 17, 105, 104, 103, 102, 50, 51, 49, 48, 1]
    if isinstance(landmarks, list):
        landmarks = np.array(landmarks)
    points = [(x[0], x[1]) for x in landmarks.astype(np.int32)[p]]
    # aimg, M = face_align.norm_crop2(img, target_face.kps, 128)
    mask = np.zeros(img.shape[:2]).astype(np.int32)
    cv2.fillConvexPoly(mask, np.array(points), color=255)

    wo, ho = list(zip(*points))
    w = np.max(wo) - np.min(wo)
    h = np.max(ho) - np.min(ho)
    mask_size = int(np.sqrt(w * h))
    k = max(mask_size // 10, 10)

    kernel = np.ones((k, k), np.uint8)
    mask = cv2.erode(mask.astype('uint8'), kernel, iterations=1)

    k = max(mask_size // 20, 5)
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    mask = cv2.GaussianBlur(mask, blur_size, 0)

    crop_shape = (max(np.min(wo), 0), max(np.min(ho), 0), min(np.max(wo), img.shape[1]), min(np.max(ho), img.shape[0]))
    landmarks = np.array(points) - np.array([np.min(wo), np.min(ho)]).tolist()
    return Image.fromarray(mask.astype(np.uint8)).crop(crop_shape), crop_shape, landmarks


arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)


def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M


def norm_crop2(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, M


def face_swap(img1, face1, img2, face2, no_seamless_copy=False):
    landmarks1, landmarks2 = get_landmarks_points106(face1), get_landmarks_points106(face2)
    aimg1, M1 = norm_crop2(img1, face1.kps, 512)
    aimg2, M2 = norm_crop2(img2, face2.kps, 512)
    tland1, tland2 = trans_points2d(np.array(landmarks1), M1), trans_points2d(np.array(landmarks2), M2)
    direction1, direction2 = get_face_direction(tland1, is_68=False), get_face_direction(tland2, is_68=False)
    if direction1 != direction2:
        tland2 = flip_points_o(tland2, aimg2.shape[1])
        aimg2 = cv2.flip(aimg2, 1)

    mask1, crop_shape1, _ = extract_face(aimg1, tland1)
    mask2, crop_shape2, _ = extract_face(aimg2, tland2)
    # miz
    crop1, crop2 = aimg1[crop_shape1[1]:crop_shape1[3], crop_shape1[0]:crop_shape1[2]], aimg2[
                                                                                        crop_shape2[1]:crop_shape2[3],
                                                                                        crop_shape2[0]:crop_shape2[2]]
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)
    crop2 = color_transfer(crop1, crop2)
    # crop1 = (crop1*(np.expand_dims(mask1,-1).repeat(3,-1)/255)).astype(np.uint8)
    # crop2 = (crop2*(np.expand_dims(mask2,-1).repeat(3,-1)/255)).astype(np.uint8)
    # SeamlessClone crop 1 in crop2
    crop2 = cv2.resize(crop2, (crop1.shape[1], crop1.shape[0]))
    if not no_seamless_copy:
        mask2 = cv2.resize(np.array(mask2), (crop1.shape[1], crop1.shape[0]))
        crop1 = cv2.seamlessClone(crop2, crop1, np.array(mask2), (crop1.shape[1] // 2, crop1.shape[0] // 2),
                                  cv2.NORMAL_CLONE)
        mask1 = np.expand_dims(mask1, -1).repeat(3, -1) / 255
    else:
        # crop1 = crop2
        mask1 = cv2.dilate(mask1,np.ones((25, 25), np.uint8),1)
        mask1 = np.expand_dims(mask1, -1).repeat(3, -1) / 255
        crop1 = crop2 * mask1 + crop1 * (1 - mask1)

    # mask1 = np.expand_dims(mask1, -1).repeat(3, -1) / 255
    aimg1[crop_shape1[1]:crop_shape1[3], crop_shape1[0]:crop_shape1[2]] = crop1 * mask1 + aimg1[
                                                                                          crop_shape1[1]:crop_shape1[3],
                                                                                          crop_shape1[0]:crop_shape1[
                                                                                              2]] * (1 - mask1)
    IM1 = cv2.invertAffineTransform(M1)
    outmask = cv2.warpAffine(np.ones(aimg1.shape[:2]), IM1, (img1.shape[1], img1.shape[0]), borderValue=0.0)
    out = cv2.warpAffine(aimg1, IM1, (img1.shape[1], img1.shape[0]), borderValue=0.0)

    k = 10
    kernel = np.ones((k, k), np.uint8)
    outmask = cv2.erode(outmask.astype('uint8'), kernel, iterations=1)
    outmask = np.expand_dims(outmask, -1).repeat(3, -1)

    return (out * outmask + img1 * (1 - outmask)).astype(np.uint8)


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def flip_points_o(points, width):
    if isinstance(points, list):
        return [(width - x, y) for (x, y) in points]
    elif isinstance(points, np.ndarray):
        return np.array([(width - x, y) for (x, y) in points])
    else:
        raise TypeError("points type error")
