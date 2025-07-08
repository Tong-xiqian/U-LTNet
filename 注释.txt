堆叠块的多个实例。

：param n_blocks：剩余块数
：param conv_op:nn。ConvNd类
：param input_channels：仅与序列中的forst块相关。这是特征的输入数量。在第一个块之后，添加残差的主路径中的特征数量为output_channels
：param output_channels：添加残差的主路径中的特征数（以及
输出的特征数量）
：param kernel_size：所有nxn（n！=1）卷积的内核大小。默认值：3x3
：param initial_stride：仅影响第一个块。所有后续区块都有步幅1
：param conv_bias：通常为False
：param norm_op:nn。批次标准Nd、实例标准Nd等
：param norm_op_kwargs：kwargs的字典。默认值为空（｛｝）
：param dropout_op:nn。DropoutNd，可以为None表示无脱落
：param dropout_op_kwargs：
：param nonlin：
：param nonlin_kwargs：
：param block:BasicBlockD或BottleneckD
：param bottleneck_channels：如果块是BottleneckD，那么我们需要知道瓶颈特征的数量。
瓶颈将首先使用1x1 conv来减少对瓶颈功能的输入，然后运行nxn（请参阅kernel_size）
关于那个（瓶颈->瓶颈）。最后，输出将投影回output_channels
（瓶颈->输出通道）与最终的1x1 conv
：param stochastic_depth_p：在残差块中应用随机深度的概率
：param crushe_exclusion：是否施加挤压和激励
：param crushe_excitationreduction_tratio：挤压和激励应减少通道的比例
与相应块的输出通道的数量相对应



nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py 54行更改网络名称
nnunetv2/utilities/get_network_from_plans.py  31行增加网络解释，37行增加网络参数

C:\Users\Admin\Desktop\train\nnUNet-master\nnunetv2\training\lr_scheduler\polylr.py中
在11行的super().__init__(optimizer, current_step if current_step is not None else -1)#pytorrch太新了，最后一个参数取消了，所以要去掉
acvl_utils版本必须0.2.0
使用pip uninstall acvl_utils
pip install acvl_utils==0.2降级

nnunetv2\inference\predict_from_raw_data.py中86行因为pytorch太新，必须加上weights_only才行，应该改为
checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                        map_location=torch.device('cpu'), weights_only=False)