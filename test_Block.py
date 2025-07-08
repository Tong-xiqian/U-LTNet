import numpy as np
import netron

from typing import Tuple, List, Union, Type
import torch.nn
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU
from dynamic_network_architectures.building_blocks.regularization import DropPath, SqueezeExcite



def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    This function is taken from the timm package (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py).

    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    This class is taken from the timm package (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py).

    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class SqueezeExcite_Res(nn.Module):
    """
    This class is taken from the timm package (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/squeeze_excite.py)
    and slightly modified so that the convolution type can be adapted.

    SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(
            self, channels, conv_op, rd_ratio=1. / 16, rd_channels=None, rd_divisor=8, add_maxpool=False,
            act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, gate_layer=nn.Sigmoid):
        super(SqueezeExcite_Res, self).__init__()
        self.add_maxpool = add_maxpool

        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)

        self.fc1 = nn.Sequential(
            conv_op(channels, rd_channels, kernel_size=1, bias=True),
            norm_layer(rd_channels) if norm_layer else nn.Identity(),
            act_layer(inplace=True)
        )

        self.fc2 = nn.Sequential(
            conv_op(rd_channels, rd_channels, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(rd_channels) if norm_layer else nn.Identity(),
            act_layer(inplace=True)
        )
        
        
        self.fc3 = nn.Sequential(
            conv_op(rd_channels, channels, kernel_size=1, bias=True),
            norm_layer(channels) if norm_layer else nn.Identity()
        )
        

        if conv_op == nn.Conv2d:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif conv_op == nn.Conv3d:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        elif conv_op == nn.Conv1d:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        else:
            raise RuntimeError("Invalid conv op: %s" % str(conv_op))

        self.fc = nn.Sequential(
            nn.Linear(channels, rd_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rd_channels, channels, bias=False),
            nn.Sigmoid()
        )

        self.gate = gate_layer()

    def forward(self, x):
        x_se = self.fc3(self.fc2(self.fc1))
        b = x_se.size(0)
        c = x_se.size(1)
        pool = self.gate(self.fc(self.fc(self.avg_pool(x_se)))).view(*x_se.shape)
        
        return self.gate(x+torch.mul(pool, x_se))


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    """
    This function is taken from the timm package (https://github.com/rwightman/pytorch-image-models/blob/b7cb8d0337b3e7b50516849805ddb9be5fc11644/timm/models/layers/helpers.py#L25)
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v



class BasicBlockD(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 # todo wideresnet?
                 ):
        """
        This implementation follows ResNet-D:

        He, Tong, et al. "Bag of tricks for image classification with convolutional neural networks."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

        The skip has an avgpool (if needed) followed by 1x1 conv instead of just a strided 1x1 conv

        :param conv_op:
        :param input_channels:
        :param output_channels:
        :param kernel_size: refers only to convs in feature extraction path, not to 1x1x1 conv in skip
        :param stride: only applies to first conv (and skip). Second conv always has stride 1
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op: only the first conv can have dropout. The second never has
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param stochastic_depth_p:
        :param squeeze_excitation:
        :param squeeze_excitation_reduction_ratio:
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        self.conv1 = ConvDropoutNormReLU(conv_op, input_channels, output_channels, kernel_size, stride, conv_bias,
                                         norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        self.conv2 = ConvDropoutNormReLU(conv_op, output_channels, output_channels, kernel_size, 1, conv_bias, norm_op,
                                         norm_op_kwargs, None, None, None, None)

        self.nonlin2 = nonlin(**nonlin_kwargs) if nonlin is not None else lambda x: x

        # Stochastic Depth
        self.apply_stochastic_depth = False if stochastic_depth_p == 0.0 else True
        if self.apply_stochastic_depth:
            self.drop_path = DropPath(drop_prob=stochastic_depth_p)

        # Squeeze Excitation
        self.apply_se = squeeze_excitation
        if self.apply_se:
            self.squeeze_excitation = SqueezeExcite(self.output_channels, conv_op,
                                                    rd_ratio=squeeze_excitation_reduction_ratio, rd_divisor=8)

        has_stride = (isinstance(stride, int) and stride != 1) or any([i != 1 for i in stride])
        requires_projection = (input_channels != output_channels)

        if has_stride or requires_projection:
            ops = []
            if has_stride:
                ops.append(get_matching_pool_op(conv_op=conv_op, adaptive=False, pool_type='avg')(stride, stride))
            if requires_projection:
                ops.append(
                    ConvDropoutNormReLU(conv_op, input_channels, output_channels, 1, 1, False, norm_op,
                                        norm_op_kwargs, None, None, None, None
                                        )
                )
            self.skip = nn.Sequential(*ops)
        else:
            self.skip = lambda x: x

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv2(self.conv1(x))
        if self.apply_stochastic_depth:
            out = self.drop_path(out)
        if self.apply_se:
            out = self.squeeze_excitation(out)
        out += residual
        return self.nonlin2(out)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        size_after_stride = [i // j for i, j in zip(input_size, self.stride)]
        # conv1
        output_size_conv1 = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        # conv2
        output_size_conv2 = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        # skip conv (if applicable)
        if (self.input_channels != self.output_channels) or any([i != j for i, j in zip(input_size, size_after_stride)]):
            assert isinstance(self.skip, nn.Sequential)
            output_size_skip = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        else:
            assert not isinstance(self.skip, nn.Sequential)
            output_size_skip = 0
        return output_size_conv1 + output_size_conv2 + output_size_skip


class BottleneckD(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 bottleneck_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16
                 ):
        """
        This implementation follows ResNet-D:

        He, Tong, et al. "Bag of tricks for image classification with convolutional neural networks."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

        The stride sits in the 3x3 conv instead of the 1x1 conv!
        The skip has an avgpool (if needed) followed by 1x1 conv instead of just a strided 1x1 conv

        :param conv_op:
        :param input_channels:
        :param output_channels:
        :param kernel_size: only affects the conv in the middle (typically 3x3). The other convs remain 1x1
        :param stride: only applies to the conv in the middle (and skip). Note that this deviates from the canonical
        ResNet implementation where the stride is applied to the first 1x1 conv. (This implementation follows ResNet-D)
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op: only the second (kernel_size) conv can have dropout. The first and last conv (1x1(x1)) never have it
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param stochastic_depth_p:
        :param squeeze_excitation:
        :param squeeze_excitation_reduction_ratio:
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bottleneck_channels = bottleneck_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        self.conv1 = ConvDropoutNormReLU(conv_op, input_channels, bottleneck_channels, 1, 1, conv_bias,
                                         norm_op, norm_op_kwargs, None, None, nonlin, nonlin_kwargs)
        self.conv2 = ConvDropoutNormReLU(conv_op, bottleneck_channels, bottleneck_channels, kernel_size, stride,
                                         conv_bias,
                                         norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        self.conv3 = ConvDropoutNormReLU(conv_op, bottleneck_channels, output_channels, 1, 1, conv_bias, norm_op,
                                         norm_op_kwargs, None, None, None, None)

        self.nonlin3 = nonlin(**nonlin_kwargs) if nonlin is not None else lambda x: x

        # Stochastic Depth
        self.apply_stochastic_depth = False if stochastic_depth_p == 0.0 else True
        if self.apply_stochastic_depth:
            self.drop_path = DropPath(drop_prob=stochastic_depth_p)

        # Squeeze Excitation
        self.apply_se = squeeze_excitation
        if self.apply_se:
            self.squeeze_excitation = SqueezeExcite_Res(self.output_channels, conv_op,
                                                    rd_ratio=squeeze_excitation_reduction_ratio, rd_divisor=8)

        has_stride = (isinstance(stride, int) and stride != 1) or any([i != 1 for i in stride])
        requires_projection = (input_channels != output_channels)

        if has_stride or requires_projection:
            ops = []
            if has_stride:
                ops.append(get_matching_pool_op(conv_op=conv_op, adaptive=False, pool_type='avg')(stride, stride))
            if requires_projection:
                ops.append(
                    ConvDropoutNormReLU(conv_op, input_channels, output_channels, 1, 1, False,
                                        norm_op, norm_op_kwargs, None, None, None, None
                                        )
                )
            self.skip = nn.Sequential(*ops)
        else:
            self.skip = lambda x: x

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv3(self.conv2(self.conv1(x)))
        if self.apply_stochastic_depth:
            out = self.drop_path(out)
        if self.apply_se:
            out = self.squeeze_excitation(out)
        print(out.size())
        print(residual.size())
        out += residual
        return self.nonlin3(out)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        size_after_stride = [i // j for i, j in zip(input_size, self.stride)]
        # conv1
        output_size_conv1 = np.prod([self.bottleneck_channels, *input_size], dtype=np.int64)
        # conv2
        output_size_conv2 = np.prod([self.bottleneck_channels, *size_after_stride], dtype=np.int64)
        # conv3
        output_size_conv3 = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        # skip conv (if applicable)
        if (self.input_channels != self.output_channels) or any([i != j for i, j in zip(input_size, size_after_stride)]):
            assert isinstance(self.skip, nn.Sequential)
            output_size_skip = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        else:
            assert not isinstance(self.skip, nn.Sequential)
            output_size_skip = 0
        return output_size_conv1 + output_size_conv2 + output_size_conv3 + output_size_skip


class Se_ResBlock(nn.Module):
    def __init__(self,
                 n_blocks: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16
                 ):
        """
        Stack multiple instances of block.

        :param n_blocks: number of residual blocks
        :param conv_op: nn.ConvNd class
        :param input_channels: only relevant for forst block in the sequence. This is the input number of features.
        After the first block, the number of features in the main path to which the residuals are added is output_channels
        :param output_channels: number of features in the main path to which the residuals are added (and also the
        number of features of the output)
        :param kernel_size: kernel size for all nxn (n!=1) convolutions. Default: 3x3
        :param initial_stride: only affects the first block. All subsequent blocks have stride 1
        :param conv_bias: usually False
        :param norm_op: nn.BatchNormNd, InstanceNormNd etc
        :param norm_op_kwargs: dictionary of kwargs. Leave empty ({}) for defaults
        :param dropout_op: nn.DropoutNd, can be None for no dropout
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param block: BasicBlockD or BottleneckD
        :param bottleneck_channels: if block is BottleneckD then we need to know the number of bottleneck features.
        Bottleneck will use first 1x1 conv to reduce input to bottleneck features, then run the nxn (see kernel_size)
        conv on that (bottleneck -> bottleneck). Finally the output will be projected back to output_channels
        (bottleneck -> output_channels) with the final 1x1 conv
        :param stochastic_depth_p: probability of applying stochastic depth in residual blocks
        :param squeeze_excitation: whether to apply squeeze and excitation or not
        :param squeeze_excitation_reduction_ratio: ratio by how much squeeze and excitation should reduce channels
        respective to number of out channels of respective block
        """
        super().__init__()
        assert n_blocks > 0, 'n_blocks must be > 0'
        assert block in [BasicBlockD, BottleneckD], 'block must be BasicBlockD or BottleneckD'
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * n_blocks
        if not isinstance(bottleneck_channels, (tuple, list)):
            bottleneck_channels = [bottleneck_channels] * n_blocks

        if block == BasicBlockD:
            blocks = nn.Sequential(
                block(conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias,
                      norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                      squeeze_excitation, squeeze_excitation_reduction_ratio),
                *[block(conv_op, output_channels[n - 1], output_channels[n], kernel_size, 1, conv_bias, norm_op,
                        norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                        squeeze_excitation, squeeze_excitation_reduction_ratio) for n in range(1, n_blocks)]
            )
        else:
            blocks = nn.Sequential(
                block(conv_op, input_channels, bottleneck_channels[0], output_channels[0], kernel_size,
                      initial_stride, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                      nonlin, nonlin_kwargs, stochastic_depth_p, squeeze_excitation, squeeze_excitation_reduction_ratio),
                *[block(conv_op, output_channels[n - 1], bottleneck_channels[n], output_channels[n], kernel_size,
                        1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                        nonlin, nonlin_kwargs, stochastic_depth_p, squeeze_excitation,
                        squeeze_excitation_reduction_ratio) for n in range(1, n_blocks)]
            )
        self.blocks = blocks
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)
        self.output_channels = output_channels[-1]

    def forward(self, x):
        return self.blocks(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output = self.blocks[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.blocks[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output


if __name__ == '__main__':
    data = torch.rand((1, 3, 40, 32))

    stx = Se_ResBlock(2, nn.Conv2d, 24, (16, 16), (3, 3), (1, 2),
                                                norm_op=nn.BatchNorm2d, nonlin=nn.ReLU, nonlin_kwargs={'inplace': True},
                                                block=BottleneckD, bottleneck_channels=3)
    model = nn.Sequential(ConvDropoutNormReLU(nn.Conv2d,
                                              3, 24, 3, 1, True, nn.BatchNorm2d, {}, None, None, nn.LeakyReLU,
                                              {'inplace': True}),
                          stx)
    import hiddenlayer as hl

    # g = hl.build_graph_with_model(model, input_tensor)

    # g = hl.build_graph(model, data,
    #                    transforms=None)
    # g.save("network_architecture.pdf")
    # del g

    # torch.onnx.export(stx, torch.rand([16, 24, 256, 256]), "testnet.onnx", opset_version=11)
    # netron.start("testnet.onnx")	# 使用netron可视化onnx模型

    torch.onnx.export(model, data, "testnet.onnx", opset_version=11)
    netron.start("testnet.onnx")	# 使用netron可视化onnx模型


    print(stx.compute_conv_feature_map_size((40, 32)))