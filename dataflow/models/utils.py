import tensorflow as tf


def leaky_relu(x, leak=0.2, name="leaky_relu"):
    """Leaky rectifier.
    Parameters
    ----------
    x : Tensor
        The tensor to apply the non-linearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Output of the non-linearity.
    """
    return tf.maximum(leak * x, x, name)
