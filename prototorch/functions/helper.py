import torch


def calculate_prototype_accuracy(y_pred, y_true, plabels):
    """Computes the accuracy of a prototype based model.
    via Winner-Takes-All rule.
    Requirement:
    y_pred.shape == y_true.shape
    unique(y_pred) in plabels
    """
    with torch.no_grad():
        idx = torch.argmin(y_pred, axis=1)
        return torch.true_divide(torch.sum(y_true == plabels[idx]),
                                 len(y_pred)) * 100


def predict_label(y_pred, plabels):
    r""" Predicts labels given a prediction of a prototype based model.
    """
    with torch.no_grad():
        return plabels[torch.argmin(y_pred, 1)]


def mixed_shape(inputs):
    if not torch.is_tensor(inputs):
        raise ValueError('Input must be a tensor.')
    else:
        int_shape = list(inputs.shape)
        # sometimes int_shape returns mixed integer types
        int_shape = [int(i) if i is not None else i for i in int_shape]
        tensor_shape = inputs.shape

        for i, s in enumerate(int_shape):
            if s is None:
                int_shape[i] = tensor_shape[i]
        return tuple(int_shape)


def equal_int_shape(shape_1, shape_2):
    if not isinstance(shape_1,
                      (tuple, list)) or not isinstance(shape_2, (tuple, list)):
        raise ValueError('Input shapes must list or tuple.')
    for shape in [shape_1, shape_2]:
        if not all([isinstance(x, int) or x is None for x in shape]):
            raise ValueError(
                'Input shapes must be list or tuple of int and None values.')

    if len(shape_1) != len(shape_2):
        return False
    else:
        for axis, value in enumerate(shape_1):
            if value is not None and shape_2[axis] not in {value, None}:
                return False
        return True


def _check_shapes(signal_int_shape, proto_int_shape):
    if len(signal_int_shape) < 4:
        raise ValueError(
            "The number of signal dimensions must be >=4. You provide: " +
            str(len(signal_int_shape)))

    if len(proto_int_shape) < 2:
        raise ValueError(
            "The number of proto dimensions must be >=2. You provide: " +
            str(len(proto_int_shape)))

    if not equal_int_shape(signal_int_shape[3:], proto_int_shape[1:]):
        raise ValueError(
            "The atom shape of signals must be equal protos. You provide: signals.shape[3:]="
            + str(signal_int_shape[3:]) + " != protos.shape[1:]=" +
            str(proto_int_shape[1:]))

    # not a sparse signal
    if signal_int_shape[1] != 1:
        if not equal_int_shape(signal_int_shape[1:2], proto_int_shape[0:1]):
            raise ValueError(
                "If the signal is not sparse, the number of prototypes must be equal in signals and "
                "protos. You provide: " + str(signal_int_shape[1]) + " != " +
                str(proto_int_shape[0]))

    return True


def _int_and_mixed_shape(tensor):
    shape = mixed_shape(tensor)
    int_shape = tuple([i if isinstance(i, int) else None for i in shape])

    return shape, int_shape
