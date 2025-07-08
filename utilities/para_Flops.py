import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn


from dynamic_network_architectures.architectures.myunet import Myunet
from dynamic_network_architectures.architectures.rnn_unet import RNN_unet
# from dynamic_network_architectures.architectures.test_unet import test_UNet
from dynamic_network_architectures.architectures.U_LTNet import U_LTNet
from thop import profile, clever_format

import torch
import json
from thop import profile, clever_format
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

# 请根据您的实际环境修改这些路径
TASK_NAME = "Dataset423_MRIIBSR18"  # 替换为您的任务名称
MODEL_PATH = r"C:\Users\Admin\Desktop\train\DATA\nnunet\DATASET\nnUNet_trained_models\Dataset423_MRIIBSR18\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_0\checkpoint_final.pth"  # 替换为您的模型路径
PLANS_PATH = r"C:\Users\Admin\Desktop\train\DATA\nnunet\DATASET\nnUNet_trained_models\Dataset423_MRIIBSR18\nnUNetTrainer__nnUNetPlans__3d_fullres\plans.json"  # plans.json 路径
DATASET_JSON_PATH = r"C:\Users\Admin\Desktop\train\DATA\nnunet\DATASET\nnUNet_trained_models\Dataset423_MRIIBSR18\nnUNetTrainer__nnUNetPlans__3d_fullres\dataset.json"  # dataset.json 路径
CONFIGURATION_NAME = "3d_fullres"  # 通常为 '2d', '3d_fullres' 或 '3d_lowres'

def load_model():
    """加载模型结构和权重"""
    # 1. 加载 plans 文件
    with open(PLANS_PATH, 'r') as f:
        plans = json.load(f)
    plans_manager = PlansManager(plans)
    
    # 2. 加载 dataset.json
    with open(DATASET_JSON_PATH, 'r') as f:
        dataset_json = json.load(f)
    
    # 3. 创建配置管理器
    configuration_dict = plans_manager.plans['configurations'][CONFIGURATION_NAME]
    configuration_manager = ConfigurationManager(configuration_dict)
    
    # 4. 获取输入通道数 (从dataset.json)
    num_input_channels = len(dataset_json['channel_names'].keys())
    
    # 5. 创建模型结构 (使用您提供的函数)
    model = get_network_from_plans(
        plans_manager,
        dataset_json,
        configuration_manager,
        num_input_channels,
        deep_supervision=True,
        calculate_metrics_flag=False  # 我们稍后会自己计算
    )
    
    # 6. 加载权重
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    
    # 处理可能的模块名前缀（如训练时使用了 DataParallel）
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def calculate_metrics(model, input_shape=(1, 1, 128, 128, 128)):
    """
    计算模型参数量和FLOPs
    input_shape: (batch, channels, depth, height, width)
    """
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    
    # 计算FLOPs
    try:
        input_tensor = torch.randn(input_shape)
        macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
        flops = macs * 2  # FLOPs ≈ 2 × MACs
    except Exception as e:
        print(f"FLOPs计算失败: {e}")
        flops = 0
    
    return total_params, flops

def get_network_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           deep_supervision: bool = True,
                           calculate_metrics_flag: bool = False,  # 新增参数
                           input_shape: tuple = (1, 1, 128, 128, 128)  # 新增参数
                           ):
    """
    修改后的函数，增加模型指标计算功能
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = configuration_manager.UNet_class_name
    mapping = {
        'PlainConvUNet': PlainConvUNet,
        'ResidualEncoderUNet': ResidualEncoderUNet,
        'Myunet': Myunet,
        'RNN_unet': RNN_unet,
        'U_LTNet': U_LTNet
    }
    kwargs = {
        'PlainConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        # ... 其他网络配置保持不变 ...
        'U_LTNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
    }
    
    assert segmentation_network_class_name in mapping.keys(), '网络架构未识别'
    network_class = mapping[segmentation_network_class_name]

    conv_or_blocks_per_stage = {
        'n_conv_per_stage'
        if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }
    
    # 创建模型
    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    
    model.apply(InitWeights_He(1e-2))
    if network_class == ResidualEncoderUNet:
        model.apply(init_last_bn_before_add_to_0)
    
    # ===== 新增指标计算部分 =====
    if calculate_metrics_flag:
        # 计算模型指标
        params, flops = calculate_model_metrics(model, input_shape)
        
        # 打印结果
        print("\n" + "="*50)
        print(f"模型架构: {segmentation_network_class_name}")
        print(f"输入尺寸: {input_shape}")
        print(f"总参数量: {params:,}")
        print(f"FLOPs: {flops:,}")
        
        # 格式化输出
        if flops > 0:
            params_m, flops_g = clever_format([(params, "params"), (flops, "flops")], "%.3f")
            print(f"参数量: {params_m}")
            print(f"FLOPs: {flops_g}")
        else:
            print(f"参数量: {params/1e6:.2f} M")
        print("="*50 + "\n")
    
    return model
# 主执行函数
def main():
    # 设置输入尺寸 (根据您的实际模型调整)
    input_shape = (1, 1, 128, 128, 128)  # (batch, channels, D, H, W)
    
    print("="*50)
    print(f"任务名称: {TASK_NAME}")
    print(f"配置名称: {CONFIGURATION_NAME}")
    print(f"输入尺寸: {input_shape}")
    print("="*50)
    
    # 加载模型
    print("正在加载模型...")
    model = load_model()
    print("模型加载成功!")
    
    # 计算指标
    print("计算参数量和FLOPs...")
    params, flops = calculate_metrics(model, input_shape)
    
    # 打印结果
    print("\n" + "="*50)
    print(f"模型架构: U_LTNet")
    print(f"总参数量: {params:,}")
    
    if flops > 0:
        print(f"FLOPs: {flops:,}")
        # 格式化输出
        params_m, flops_g = clever_format([(params, "params"), (flops, "flops")], "%.3f")
        print(f"\n参数量: {params_m}")
        print(f"FLOPs: {flops_g}")
        print(f"参数量: {params/1e6:.2f} M")
        print(f"FLOPs: {flops/1e9:.2f} G")
    else:
        print("FLOPs计算失败，仅显示参数量")
        print(f"参数量: {params/1e6:.2f} M")
    print("="*50)

if __name__ == "__main__":
    # 确保这些路径正确
    import os
    print(f"Plans 文件路径: {PLANS_PATH} - 存在: {os.path.exists(PLANS_PATH)}")
    print(f"Dataset JSON 路径: {DATASET_JSON_PATH} - 存在: {os.path.exists(DATASET_JSON_PATH)}")
    print(f"模型路径: {MODEL_PATH} - 存在: {os.path.exists(MODEL_PATH)}")
    
    main()