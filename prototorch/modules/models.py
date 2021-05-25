import torch
from prototorch.functions.distances import (euclidean_distance_matrix,
                                            tangent_distance)
from prototorch.functions.helper import _check_shapes, _int_and_mixed_shape
from prototorch.functions.normalization import orthogonalization
from prototorch.modules.prototypes import Prototypes1D
from torch import nn


class GTLVQ(nn.Module):
    r""" Generalized Tangent Learning Vector Quantization

    Parameters
    ----------
    num_classes: int
        Number of classes of the given classification problem.

    subspace_data: torch.tensor of shape (n_batch,feature_dim,feature_dim)
        Subspace data for the point approximation, required

    prototype_data: torch.tensor of shape (n_init_data,feature_dim) (optional)
        prototype data for initalization of the prototypes used in GTLVQ.

    subspace_size: int (default=256,optional)
        Subspace dimension of the Projectors. Currently only supported
        with tagnent_projection_type=global.

    tangent_projection_type: string
        Specifies the tangent projection type
        options:    local
                    local_proj
                    global
        local: computes the tangent distances without emphasizing projected
        data. Only distances are available
        local_proj: computs tangent distances and returns the projected data
        for further use. Be careful: data is repeated by number of prototypes
        global: Number of subspaces is set to one and every prototypes
        uses the same.

    prototypes_per_class: int (default=2,optional)
    Number of prototypes per class

    feature_dim: int (default=256)
    Dimensionality of the feature space specified as integer.
    Prototype dimension.

    Notes
    -----
    The GTLVQ [1] is a prototype-based classification learning model. The
    GTLVQ uses the Tangent-Distances for a local point approximation
    of an assumed data manifold via prototypial representations.

    The GTLVQ requires subspace projectors for transforming the data
    and prototypes into the affine subspace. Every prototype is
    equipped with a specific subpspace and represents a point
    approximation of the assumed manifold.

    In practice prototypes and data are projected on this manifold
    and pairwise euclidean distance computes.

    References
    ----------
    .. [1] Saralajew, Sascha; Villmann, Thomas: Transfer learning
    in classification based on manifolc. models and its relation
    to tangent metric learning. In: 2017 International Joint
    Conference on Neural Networks (IJCNN).
    Bd. 2017-May : IEEE, 2017, S. 1756–1765
    """
    def __init__(
        self,
        num_classes,
        subspace_data=None,
        prototype_data=None,
        subspace_size=256,
        tangent_projection_type="local",
        prototypes_per_class=2,
        feature_dim=256,
    ):
        super(GTLVQ, self).__init__()

        self.num_protos = num_classes * prototypes_per_class
        self.subspace_size = feature_dim if subspace_size is None else subspace_size
        self.feature_dim = feature_dim

        if subspace_data is None:
            raise ValueError("Init Data must be specified!")

        self.tpt = tangent_projection_type
        with torch.no_grad():
            if self.tpt == "local" or self.tpt == "local_proj":
                self.init_local_subspace(subspace_data)
            elif self.tpt == "global":
                self.init_gobal_subspace(subspace_data, subspace_size)
            else:
                self.subspaces = None

        # Hypothesis-Margin-Classifier
        self.cls = Prototypes1D(
            input_dim=feature_dim,
            prototypes_per_class=prototypes_per_class,
            num_classes=num_classes,
            prototype_initializer="stratified_mean",
            data=prototype_data,
        )

    def forward(self, x):
        # Tangent Projection
        if self.tpt == "local_proj":
            x_conform = (x.unsqueeze(1).repeat_interleave(self.num_protos,
                                                          1).unsqueeze(2))
            dis, proj_x = self.local_tangent_projection(x_conform)

            proj_x = proj_x.reshape(x.shape[0] * self.num_protos,
                                    self.feature_dim)
            return proj_x, dis
        elif self.tpt == "local":
            x_conform = (x.unsqueeze(1).repeat_interleave(self.num_protos,
                                                          1).unsqueeze(2))
            dis = tangent_distance(x_conform, self.cls.prototypes,
                                   self.subspaces)
        elif self.tpt == "gloabl":
            dis = self.global_tangent_distances(x)
        else:
            dis = (x @ self.cls.prototypes.T) / (
                torch.norm(x, dim=1, keepdim=True) @ torch.norm(
                    self.cls.prototypes, dim=1, keepdim=True).T)
        return dis

    def init_gobal_subspace(self, data, num_subspaces):
        _, _, v = torch.svd(data)
        subspace = (torch.eye(v.shape[0]) - (v @ v.T)).T
        subspaces = subspace[:, :num_subspaces]
        self.subspaces = (torch.nn.Parameter(
            subspaces).clone().detach().requires_grad_(True))

    def init_local_subspace(self, data):
        _, _, v = torch.svd(data)
        inital_projector = (torch.eye(v.shape[0]) - (v @ v.T)).T
        subspaces = inital_projector.unsqueeze(0).repeat_interleave(
            self.num_protos, 0)
        self.subspaces = (torch.nn.Parameter(
            subspaces).clone().detach().requires_grad_(True))

    def global_tangent_distances(self, x):
        # Tangent Projection
        x, projected_prototypes = (
            x @ self.subspaces,
            self.cls.prototypes @ self.subspaces,
        )
        # Euclidean Distance
        return euclidean_distance_matrix(x, projected_prototypes)

    def local_tangent_projection(self, signals):
        # Note: subspaces is always assumed as transposed and must be orthogonal!
        # shape(signals): batch x proto_number x channels x dim1 x dim2 x ... x dimN
        # shape(protos): proto_number x dim1 x dim2 x ... x dimN
        # shape(subspaces): (optional [proto_number]) x prod(dim1 * dim2 * ... * dimN)  x prod(projected_atom_shape)
        # subspace should be orthogonalized
        # Origin Source Code
        # Origin Author:
        protos = self.cls.prototypes
        subspaces = self.subspaces
        signal_shape, signal_int_shape = _int_and_mixed_shape(signals)
        _, proto_int_shape = _int_and_mixed_shape(protos)

        # check if the shapes are correct
        _check_shapes(signal_int_shape, proto_int_shape)

        # Tangent Data Projections
        projected_protos = torch.bmm(protos.unsqueeze(1), subspaces).squeeze(1)
        data = signals.squeeze(2).permute([1, 0, 2])
        projected_data = torch.bmm(data, subspaces)
        projected_data = projected_data.permute([1, 0, 2]).unsqueeze(1)
        diff = projected_data - projected_protos
        projected_diff = torch.reshape(
            diff, (signal_shape[1], signal_shape[0], signal_shape[2]) +
            signal_shape[3:])
        diss = torch.norm(projected_diff, 2, dim=-1)
        return diss.permute([1, 0, 2]).squeeze(-1), projected_data.squeeze(1)

    def get_parameters(self):
        return {
            "params": self.cls.prototypes,
        }, {
            "params": self.subspaces
        }

    def orthogonalize_subspace(self):
        if self.subspaces is not None:
            with torch.no_grad():
                ortho_subpsaces = (orthogonalization(self.subspaces)
                                   if self.tpt == "global" else
                                   torch.nn.init.orthogonal_(self.subspaces))
                self.subspaces.copy_(ortho_subpsaces)
