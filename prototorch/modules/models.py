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
        self.num_protos_class = prototypes_per_class
        self.subspace_size = feature_dim if subspace_size is None else subspace_size
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.cls = Prototypes1D(
            input_dim=feature_dim,
            prototypes_per_class=prototypes_per_class,
            nclasses=num_classes,
            prototype_initializer="stratified_mean",
            data=prototype_data,
        )

        if subspace_data is None:
            raise ValueError("Init Data must be specified!")

        self.tpt = tangent_projection_type
        with torch.no_grad():
            if self.tpt == "local":
                self.init_local_subspace(subspace_data, subspace_size, self.num_protos)
            elif self.tpt == "global":
                self.init_gobal_subspace(subspace_data, subspace_size)
            else:
                self.subspaces = None

    def forward(self, x):
        if self.tpt == "local":
            dis = self.local_tangent_distances(x)
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
        self.subspaces = nn.Parameter(subspaces,requires_grad=True)

    def init_local_subspace(self, data,num_subspaces,num_protos):
        data = data -  torch.mean(data,dim=0)      
        _,_,v = torch.svd(data,some=False)
        v = v[:,:num_subspaces]
        subspaces = v.unsqueeze(0).repeat_interleave(num_protos,0)
        self.subspaces = nn.Parameter(subspaces,requires_grad=True) 

    def global_tangent_distances(self, x):
        # Tangent Projection
        x, projected_prototypes = (
            x @ self.subspaces,
            self.cls.prototypes @ self.subspaces,
        )
        # Euclidean Distance
        return euclidean_distance_matrix(x, projected_prototypes)

    def local_tangent_distances(self, x):

        # Tangent Distance
        x = x.unsqueeze(1).expand(x.size(0), self.cls.prototypes.size(0), x.size(-1))
        protos = self.cls.prototypes.unsqueeze(0).expand(x.size(0), self.cls.prototypes.size(0), x.size(-1))
        projectors =torch.eye(self.subspaces.shape[-2],device=x.device) - torch.bmm(self.subspaces,self.subspaces.permute([0,2,1]))
        diff = (x - protos)
        diff = diff.permute([1, 0, 2]) 
        diff = torch.bmm(diff, projectors) 
        diff = torch.norm(diff,2,dim=-1).T
        return diff 

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
