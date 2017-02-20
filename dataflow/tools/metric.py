import sys

import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve

from dataflow.loader.io import YuvReader, imread

BATCH_SIZE = 10

"""Python implementation of PSNR."""


def cal_img_PSNR(org_image, test_image):
    if isinstance(org_image, str):
        org_image = imread(org_image)
    if isinstance(test_image, str):
        test_image = imread(test_image)

    mse = np.mean(np.square(org_image - test_image))
    psnr = np.clip(np.multiply(np.log10(255. * 255. / mse[mse > 0.]), 10.), 0., 99.99)
    return psnr


def cal_yuv_PSNR(org_yuv_path, test_yuv_path, width=None, height=None, n_frames=None, v_format='yuv420'):
    """Return the PSNR score between org_yuv and test_yuv

    Arguments:
        org_yuv_path: Path of Original YUV
        test_yuv_path: Path of Reconstructed YUV
        width: width of org_yuv and test_yuv; if None, the width value will be inferred through yuv filename
        height: same as width, the height of org_yuv and test_yuv
        n_frames: number of frames need to calculate PSNR score
        v_format: YUV format: yuv420, yuv400, and yuv444

    Returns:
        list of PSNR score between org_yuv and test_yuv, which includes (Y_PSNR, U_PSNR, V_PSNR)

    """
    org_yuv_reader = YuvReader(org_yuv_path, y_h=height, y_w=width, v_format=v_format)
    test_yuv_reader = YuvReader(test_yuv_path, y_h=height, y_w=width, v_format=v_format)

    sum_psnr_y = sum_psnr_u = sum_psnr_v = 0.
    # num of frames to read every time
    frame_count = 0
    while True:
        batch_org_y, batch_org_u, batch_org_v = org_yuv_reader.read(n_frames=BATCH_SIZE, yuv_is_tuple=True)
        batch_test_y, batch_test_u, batch_test_v = test_yuv_reader.read(n_frames=BATCH_SIZE, yuv_is_tuple=True)

        eof1 = batch_org_y is None or batch_test_y is None
        eof2 = n_frames is not None and frame_count >= n_frames
        if eof1 or eof2:
            break

        frame_count += batch_org_y.shape[0]
        mse_y = np.mean(np.square(batch_org_y - batch_test_y), axis=(1, 2, 3))
        mse_y[mse_y == 0.] = sys.float_info.epsilon

        psnr_y = np.clip(np.multiply(np.log10(255. * 255. / mse_y[mse_y > 0.]), 10.), 0., 99.99)
        sum_psnr_y += np.sum(psnr_y)

        print('average PSNR_Y of first %d frames: %f' % (frame_count, sum_psnr_y / frame_count))

        if v_format != 'yuv400':
            mse_u = np.mean(np.square(batch_org_u - batch_test_u), axis=(1, 2, 3))
            mse_v = np.mean(np.square(batch_org_v - batch_test_v), axis=(1, 2, 3))

            mse_u[mse_u == 0.] = sys.float_info.epsilon
            mse_v[mse_v == 0.] = sys.float_info.epsilon

            psnr_u = np.clip(np.multiply(np.log10(255. * 255. / mse_u[mse_u > 0.]), 10.), 0., 99.99)
            psnr_v = np.clip(np.multiply(np.log10(255. * 255. / mse_v[mse_v > 0.]), 10.), 0., 99.99)

            sum_psnr_u += np.sum(psnr_u)
            sum_psnr_v += np.sum(psnr_v)

    avg_psnr_y = sum_psnr_y / frame_count
    avg_psnr_u = sum_psnr_u / frame_count
    avg_psnr_v = sum_psnr_v / frame_count
    return avg_psnr_y, avg_psnr_u, avg_psnr_v


"""Python implementation of MS-SSIM."""


def cal_img_MSSSIM(org_image, test_image):
    if isinstance(org_image, str):
        org_image = imread(org_image)
    if isinstance(test_image, str):
        test_image = imread(test_image)

    org_image = org_image[None, ...] if org_image.ndim == 3 else org_image
    test_image = test_image[None, ...] if test_image.ndim == 3 else test_image

    return _MultiScaleSSIM(org_image, test_image, max_val=255)


def cal_yuv_MSSSIM(org_yuv_path, test_yuv_path, width=None, height=None, n_frames=None, v_format='yuv420'):
    """Return the PSNR score between org_yuv and test_yuv

    Arguments:
        org_yuv_path: Path of Original YUV
        test_yuv_path: Path of Reconstructed YUV
        width: width of org_yuv and test_yuv; if None, the width value will be inferred through yuv filename
        height: same as width, the height of org_yuv and test_yuv
        n_frames: number of frames need to calculate PSNR score
        v_format: YUV format: yuv420, yuv400, and yuv444

    Returns:
        list of PSNR score between org_yuv and test_yuv, which includes (Y_PSNR, U_PSNR, V_PSNR)
    """
    org_yuv_reader = YuvReader(org_yuv_path, y_h=height, y_w=width, v_format=v_format)
    test_yuv_reader = YuvReader(test_yuv_path, y_h=height, y_w=width, v_format=v_format)

    sum_msssim_y = sum_msssim_u = sum_msssim_v = 0.
    # num of frames to read every time
    frame_count = 0
    while True:
        batch_org_y, batch_org_u, batch_org_v = org_yuv_reader.read(n_frames=10, yuv_is_tuple=True)
        batch_test_y, batch_test_u, batch_test_v = test_yuv_reader.read(n_frames=10, yuv_is_tuple=True)

        eof1 = batch_org_y is None or batch_test_y is None
        eof2 = n_frames is not None and frame_count >= n_frames
        if eof1 or eof2:
            break

        frame_count += batch_org_y.shape[0]

        avg_msssim_y = _MultiScaleSSIM(batch_org_y, batch_test_y, max_val=255)
        sum_msssim_y += avg_msssim_y * batch_org_y.shape[0]

        print('average MSSSIM_Y of first %d frames: %f' % (frame_count, avg_msssim_y))

        if v_format != 'yuv400':
            avg_msssim_u = _MultiScaleSSIM(batch_org_u, batch_test_u, max_val=255)
            avg_msssim_v = _MultiScaleSSIM(batch_org_v, batch_test_v, max_val=255)

            sum_msssim_u += avg_msssim_u * frame_count
            sum_msssim_v += avg_msssim_v * frame_count

    avg_psnr_y = sum_msssim_y / frame_count
    avg_psnr_u = sum_msssim_u / frame_count
    avg_psnr_v = sum_msssim_v / frame_count
    return avg_psnr_y, avg_psnr_u, avg_psnr_v


def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
    """Return the Structural Similarity Map between `img1` and `img2`.

    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    Arguments:
      img1: Numpy array holding the first RGB image batch.
      img2: Numpy array holding the second RGB image batch.
      max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Size of blur kernel to use (will be reduced for small images).
      filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
      k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
      k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).

    Returns:
      Pair containing the mean SSIM and contrast sensitivity between `img1` and
      `img2`.

    Raises:
      RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                           img1.shape, img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
    cs = np.mean(v1 / v2)
    return ssim, cs


def _MultiScaleSSIM(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5,
                    k1=0.01, k2=0.03, weights=None):
    """Return the MS-SSIM score between `img1` and `img2`.

    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf

    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    Arguments:
      img1: Numpy array holding the first RGB image batch.
      img2: Numpy array holding the second RGB image batch.
      max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Size of blur kernel to use (will be reduced for small images).
      filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
      k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
      k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).
      weights: List of weights for each level; if none, use five levels and the
        weights from the original paper.

    Returns:
      MS-SSIM score between `img1` and `img2`.

    Raises:
      RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                           img1.shape, img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
    weights = np.array(weights if weights else
                       [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
    im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
    mssim = np.array([])
    mcs = np.array([])
    for _ in range(levels):
        ssim, cs = _SSIMForMultiScale(
            im1, im2, max_val=max_val, filter_size=filter_size,
            filter_sigma=filter_sigma, k1=k1, k2=k2)
        mssim = np.append(mssim, ssim)
        mcs = np.append(mcs, cs)
        filtered = [convolve(im, downsample_filter, mode='reflect')
                    for im in [im1, im2]]
        im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
    return (np.prod(mcs[0:levels - 1] ** weights[0:levels - 1]) *
            (mssim[levels - 1] ** weights[levels - 1]))
