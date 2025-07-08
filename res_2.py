from typing import Tuple, List, Union, Type
import torch.nn
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.residual import BottleneckD, BasicBlockD

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU
from dynamic_network_architectures.building_blocks.regularization import DropPath, SqueezeExcite
import numpy as np

from dynamic_network_architectures.building_blocks.kan import KANLinear


# 这里就是我们定义的特殊算子DebugOP
class DebugOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, name):  # 这个DebugOp算子将输入x直接返回，不做任何云算，插入到网络中也就不改变网络结构
        return x
    @staticmethod
    def symbolic(g, x, name):
        return g.op("my::Debug", x, name_s=name)
# 获取自定义算子的调用接口(用法上相当于实例化)，后面就可以用debug_apply(x,name进行使用)，在不同的地方可以传入不同的name
debug_apply = DebugOp.apply





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
            self, 
            channels, 
            conv_op, 
            rd_ratio=1. / 16, 
            rd_channels=None, 
            rd_divisor=8, 
            act_layer=nn.ReLU, 
            norm_op: Union[None, Type[nn.Module]] = None,
            norm_op_kwargs: dict = None,
            dropout_op: Union[None, Type[_DropoutNd]] = None,
            dropout_op_kwargs: dict = None,
            nonlin: Union[None, Type[torch.nn.Module]] = None,
            nonlin_kwargs: dict = None,
            gate_layer=nn.Sigmoid):
        super(SqueezeExcite_Res, self).__init__()

        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        
        self.fc1 = ConvDropoutNormReLU(conv_op, channels, rd_channels, 1, 1, True,
                                        norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        
        self.fc2 = ConvDropoutNormReLU(conv_op, rd_channels, rd_channels, 3, 1, True,
                                         norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)

        self.fc3 = ConvDropoutNormReLU(conv_op, rd_channels, channels, 1, 1, True,
                                         norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)  

        self.avg_pool = get_matching_pool_op(conv_op=conv_op, adaptive=True, pool_type='avg')(1)


        self.fc = nn.Sequential(
            nn.Linear(channels, rd_channels, bias=False),
            act_layer(inplace=True),
            nn.Linear(rd_channels, channels, bias=False)
        )

        self.gate = gate_layer()

    def forward(self, x):

        x_se = self.fc3(self.fc2(self.fc1(x)))
            
        avg = self.avg_pool(x_se).squeeze()
        res = self.gate(self.fc(avg.unsqueeze(0) if avg.dim() == 1 else avg))
    
        for _ in range(x_se.dim() - res.dim()):
            res = res.unsqueeze(-1)

        out = x + res * x_se
        return out


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


class my_BottleneckD(nn.Module):
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
                 squeeze_excitation: bool = True,
                 squeeze_excitation_reduction_ratio: float = 1. / 16
                 ):

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

        self.nonlin3 = nonlin(**nonlin_kwargs) if nonlin is not None else lambda x: x

        # Squeeze Excitation
        
        self.squeeze_excitation = SqueezeExcite_Res(self.output_channels, conv_op, rd_ratio=squeeze_excitation_reduction_ratio, rd_divisor=8, norm_op=norm_op,
                                                    norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                                                    nonlin=nonlin, nonlin_kwargs=nonlin_kwargs)
        
        self.cat = ConvDropoutNormReLU(conv_op, 2*self.input_channels, self.output_channels, 1, 1, True,
                                         norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)  
        
        self.avg_pooling = get_matching_pool_op(conv_op=conv_op, adaptive=True, pool_type='avg')(1)

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

        self.lstm_function = False

        self.dropout = nn.Dropout(0.2)



        


    def lstm(self, hiddens, hidden_size):
        # 这里是初始化lstm函数的逻辑
        self.seq_len, self.batch_size, self.inp_dim = hiddens.size()
        self.lstm_fun = nn.LSTM(self.inp_dim, hidden_size, num_layers=2).cuda()
        # self.h0 = torch.nn.Parameter(torch.zeros(2, batch_size, 20).cuda())
        # self.c0 = torch.nn.Parameter(torch.zeros(2, batch_size, 20).cuda())
        # print("Loading Lstm ok")
        self.lstm_linear = nn.Linear(hidden_size, self.inp_dim).cuda()
        
        

        self.lstm_function = True

    def forward(self, x, h, c,hidden_size):

        out = self.skip(x)# x:b,c,w,h,z

        out = self.squeeze_excitation(out)
        
        new_shape = (out.size(1), out.size(0), -1)
        new_hidden = out.view(new_shape)      

        if not self.lstm_function:
            self.lstm(new_hidden, hidden_size)
            

        # print("h:{}\n".format(h.size()))

        new_hidden, (h, c) = self.lstm_fun(new_hidden, (h, c))

        new_hidden = self.lstm_linear(new_hidden).view(out.size())

        out = out + new_hidden
        # out = self.cat(torch.cat((out, new_hidden), dim=1))
        

        out = self.dropout(out)


        out=self.nonlin3(out)

        return out, h , c

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
                 squeeze_excitation: bool = True,
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
        
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * n_blocks
        if not isinstance(bottleneck_channels, (tuple, list)):
            bottleneck_channels = [bottleneck_channels] * n_blocks

        blocks = nn.Sequential(
            my_BottleneckD(conv_op, input_channels, bottleneck_channels[0], output_channels[0], kernel_size,
                    initial_stride, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                    nonlin, nonlin_kwargs, stochastic_depth_p, squeeze_excitation, squeeze_excitation_reduction_ratio)
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
    # import hiddenlayer as hl

    # g = hl.build_graph(model, data,
    #                    transforms=None)
    # g.save("network_architecture.pdf")
    # del g


    import netron

    torch.onnx.export(stx, torch.rand([16, 24, 256, 256]), "testnet.onnx", opset_version=11)
    netron.start("testnet.onnx")	# 使用netron可视化onnx模型

    # torch.onnx.export(model, data, "testnet.onnx", opset_version=11)
    # netron.start("testnet.onnx")	# 使用netron可视化onnx模型

    print(stx.compute_conv_feature_map_size((40, 32)))