import torch as T
import torch.nn.functional as F
from torch_cluster import knn_graph

import utils
from vgg import VGG19


class StyleTransferLosses(VGG19):
    def __init__(self, weight_file, content_img: T.Tensor, style_img: T.Tensor, content_layers, style_layers,
                 scale_by_y=False, content_weights=None, style_weights=None):
        super(StyleTransferLosses, self).__init__(weight_file)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.scale_by_y = scale_by_y

        content_weights = content_weights if content_weights is not None else [1.] * len(self.content_layers)
        style_weights = style_weights if style_weights is not None else [1.] * len(self.style_layers)
        self.content_weights = {}
        self.style_weights = {}

        content_features = content_img
        style_features = style_img
        self.content_features = {}
        self.style_features = {}
        if scale_by_y:
            self.weights = {}

        i, j = 0, 0
        self.to(content_img.device)
        with T.no_grad():
            for name, layer in self.named_children():
                content_features = layer(content_features)
                style_features = layer(style_features)
                if name in content_layers:
                    self.content_features[name] = content_features
                    if scale_by_y:
                        self.weights[name] = T.minimum(content_features, T.sigmoid(content_features))

                    self.content_weights[name] = content_weights[i]
                    i += 1

                if name in style_layers:
                    self.style_features[name] = utils.gram_matrix(style_features)
                    self.style_weights[name] = style_weights[j]
                    j += 1

    def forward(self, input):
        content_loss, style_loss = 0., 0.
        features = input
        for name, layer in self.named_children():
            features = layer(features)
            if name in self.content_layers:
                loss = features - self.content_features[name]
                if self.scale_by_y:
                    loss *= self.weights[name]

                content_loss += (T.mean(loss ** 2) * self.content_weights[name])

            if name in self.style_layers:
                loss = F.mse_loss(self.style_features[name], utils.gram_matrix(features), reduction='sum')
                style_loss += (loss * self.style_weights[name])

        return content_loss, style_loss


def total_variation_loss(location: T.Tensor, curve_s: T.Tensor, curve_e: T.Tensor, K=10):
    se_vec = curve_e - curve_s
    x_nn_idcs = knn_graph(location, k=K)[0]
    x_sig_nns = se_vec[x_nn_idcs].view(*((se_vec.shape[0], K) + se_vec.shape[1:]))
    dist_to_centroid = T.mean(T.sum((utils.projection(x_sig_nns) - utils.projection(se_vec)[..., None, :]) ** 2, dim=-1))
    return dist_to_centroid


def curvature_loss(curve_s: T.Tensor, curve_e: T.Tensor, curve_c: T.Tensor):
    v1 = curve_s - curve_c
    v2 = curve_e - curve_c
    dist_se = T.norm(curve_e - curve_s, dim=-1) + 1e-6
    return T.mean(T.norm(v1 + v2, dim=-1) / dist_se)


def tv_loss(x):
    diff_i = T.mean((x[..., :, 1:] - x[..., :, :-1]) ** 2)
    diff_j = T.mean((x[..., 1:, :] - x[..., :-1, :]) ** 2)
    loss = diff_i + diff_j
    return loss
    
def length_loss(curve_s: T.Tensor, curve_e: T.Tensor, curve_c: T.Tensor, desired_length):

    s_x = curve_s[..., 0:1]
    s_y = curve_s[..., 1:2]
    e_x = curve_e[..., 0:1]
    e_y = curve_e[..., 1:2]
    c_x = curve_c[..., 0:1]
    c_y = curve_c[..., 1:2]
    
    A_0 = s_x + e_x - 2.0*c_x
    B_0 = -2.0 * s_x + 2.0 * c_x

    
    A_1 = T.square(A_0)
    B_1 = 2 * T.mul(B_0, A_0)
    C_1 = T.square(B_0) + 1
    
    U_1 = T.div((2 * A_1 + B_1), (2 * A_1))
    V_1 = T.sqrt(T.div((2.0 * C_1 + A_1), A_1))
    
    TT_1 = T.div(4*T.mul(A_1, C_1) - T.square(B_1), 4 * T.square(A_1))
    
    U_2 = T.div(B_1, 2 * A_1)
    V_2 = T.sqrt(T.div(C_1, A_1))
    
    
    INT_1 = U_1 * V_1 + TT_1 * T.log(V_1 + U_1)
    
    INT_2 = U_2 * V_2 + TT_1 * T.log(V_2 + U_2)
    
    
    length = T.sqrt(A_1) * 0.5 * (INT_1 - INT_2)

    return T.mean(T.square(length - desired_length))
    
def width_loss(width: T.Tensor, desired_width):
    return T.mean(T.square(width - desired_width))