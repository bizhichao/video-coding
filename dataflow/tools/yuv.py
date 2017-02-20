import numpy as np


def upsample_chroma(chroma_nhwc):
    """upsample from (H, W, C) to (2H, 2W, C), or from (N, H, W, C) to (N, 2H, 2W, C)"""
    ndim = chroma_nhwc.ndim
    dim_h = ndim - 3
    dim_w = ndim - 2
    upsampled_chroma = np.repeat(chroma_nhwc, 2, axis=dim_h)
    upsampled_chroma = np.repeat(upsampled_chroma, 2, axis=dim_w)
    return upsampled_chroma


def downsample_chroma(chroma_nhwc):
    if chroma_nhwc.ndim == 3:
        return chroma_nhwc[::2, ::2, :]
    elif chroma_nhwc.ndim > 3:
        return chroma_nhwc[..., ::2, ::2, :]
    else:
        raise NotImplementedError('Currently not support 2-dim or 1-dim np.ndarray')


def rgb2yuv(img, version):
    if version == 'bt601':
        return _rgb2yuv_bt601(img)
    elif version == 'jpeg':
        return _rgb2yuv_jpeg(img)
    elif version == 'hdtv':
        return _rgb2yuv_hdtv(img)
    else:
        raise NotImplementedError('The specified version %s is not implemented' % version)


def yuv2rgb(img, version):
    if version == 'bt601':
        return _yuv2rgb_bt601(img)
    elif version == 'jpeg':
        return _yuv2rgb_jpeg(img)
    elif version == 'hdtv':
        return _yuv2rgb_hdtv(img)
    else:
        raise NotImplementedError('The specified version %s is not implemented' % version)


def _rgb2yuv_bt601(img):
    assert img.dtype == 'uint8', 'rgb2yuv only implemented for uint8 arrays'
    rgb = np.ndarray(img.shape)  # float64
    rgb[:, :, :] = img[:, :, :]

    # transform matrix : ITU-R BT.601 version (SDTV)
    transform_matrix = np.array([[0.257, 0.504, 0.098],
                                 [-0.148, -0.291, 0.439],
                                 [0.439, -0.368, -0.071]])

    # transform and clip
    yuv = np.dot(rgb, transform_matrix.T)
    yuv[:, :, 0] = (yuv[:, :, 0] + 16.).clip(16, 235)
    yuv[:, :, 1:] = (yuv[:, :, 1:] + 128.).clip(16, 240)

    return yuv.astype('uint8')


def _yuv2rgb_bt601(img):
    assert img.dtype == 'uint8', 'yuv2rgb only implemented for uint8 arrays'
    # better clip input to the valid range just to be on the safe side
    yuv = np.ndarray(img.shape)  # float64
    yuv[:, :, 0] = img[:, :, 0].clip(16, 235).astype(yuv.dtype) - 16
    yuv[:, :, 1:] = img[:, :, 1:].clip(16, 240).astype(yuv.dtype) - 128

    # transform matrix : ITU-R BT.601 version (SDTV)
    transform_matrix = np.array([[1.164, 0., 1.596],
                                 [1.164, -0.392, -0.813],
                                 [1.164, 2.017, 0.]])

    ## ITU-R BT.709 version (HDTV)
    #  transform_matrix = array([[1.164,     0.,  1.793],
    #             [1.164, -0.213, -0.533],
    #             [1.164,  2.112,     0.]])

    rgb = np.dot(yuv, transform_matrix.T)
    return rgb.clip(0, 255).astype('uint8')


def _rgb2yuv_jpeg(img):
    assert img.dtype == 'uint8', 'rgb2yuv only implemented for uint8 arrays'
    rgb = np.ndarray(img.shape)  # float64
    rgb[:, :, :] = img[:, :, :]

    # transform matrix : JPEG
    transform_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.500],
                                 [0.500, -0.419, -0.081]])

    # transform and clip
    yuv = np.dot(rgb, transform_matrix.T)
    yuv[:, :, 0] = (yuv[:, :, 0]).clip(0, 255)
    yuv[:, :, 1:] = (yuv[:, :, 1:] + 128.).clip(0, 255)

    return yuv.astype('uint8')


def _yuv2rgb_jpeg(img):
    assert img.dtype == 'uint8', 'yuv2rgb only implemented for uint8 arrays'
    # better clip input to the valid range just to be on the safe side
    yuv = np.ndarray(img.shape)  # float64
    yuv[:, :, 0] = img[:, :, 0].clip(16, 235).astype(yuv.dtype)
    yuv[:, :, 1:] = img[:, :, 1:].clip(16, 240).astype(yuv.dtype) - 128

    # transform matrix : JPEG
    transform_matrix = np.array([[1.0, 0., 1.400],
                                 [1.0, -0.343, -0.711],
                                 [1.0, 1.765, 0.0]])

    rgb = np.dot(yuv, transform_matrix.T)
    return rgb.clip(0, 255).astype('uint8')


def _rgb2yuv_hdtv(img):
    assert img.dtype == 'uint8', 'rgb2yuv only implemented for uint8 arrays'
    rgb = np.ndarray(img.shape)  # float64
    rgb[:, :, :] = img[:, :, :]

    # transform matrix : ITU-R BT.709 version (HDTV)
    transform_matrix = np.array([[0.183, 0.614, 0.062],
                                 [-0.101, -0.339, 0.439],
                                 [0.439, -0.399, -0.040]])

    # transform and clip
    yuv = np.dot(rgb, transform_matrix.T)
    yuv[:, :, 0] = (yuv[:, :, 0] + 16.).clip(16, 235)
    yuv[:, :, 1:] = (yuv[:, :, 1:] + 128.).clip(16, 240)

    return yuv.astype('uint8')


def _yuv2rgb_hdtv(img):
    assert img.dtype == 'uint8', 'yuv2rgb only implemented for uint8 arrays'
    # better clip input to the valid range just to be on the safe side
    yuv = np.ndarray(img.shape)  # float64
    yuv[:, :, 0] = img[:, :, 0].clip(16, 235).astype(yuv.dtype) - 16
    yuv[:, :, 1:] = img[:, :, 1:].clip(16, 240).astype(yuv.dtype) - 128

    # transform matrix : ITU-R BT.709 version (HDTV)
    transform_matrix = np.array([[1.164, 0., 1.793],
                                 [1.164, -0.213, -0.533],
                                 [1.164, 2.112, 0.]])

    rgb = np.dot(yuv, transform_matrix.T)
    return rgb.clip(0, 255).astype('uint8')
