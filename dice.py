# 计算三维下各种指标
from __future__ import absolute_import, print_function

import pandas as pd
import GeodisTK 
import argparse
import numpy as np
from scipy import ndimage
from medpy import metric



import os
import nibabel as nib


def parse_args():
    #建立解释器
    parser = argparse.ArgumentParser(description='data') #建立解释器

    # 模型名称
    parser.add_argument('-seg', '--seg_path', default='result/20_20/2d/0', type=str, metavar='name',
                        help='data name')
    #：'-'表示简称，'--'表示全称
    #nargs：应该读取的命令行参数个数，可以是具体的数字
    #action：表示若触发则值变成store_下划线后面的值，未触发则为相反值，store_const表示触发才有const值，默认优先级低于default，但是一旦触发，优先级高于default
    #const - action 和 nargs 所需要的常量值。
    #default - 不指定参数时的默认值。
    #type - 命令行参数应该被转换成的类型。
    #choices - 参数可允许的值的一个容器。
    #required - 可选参数是否可以省略 (仅针对可选参数)。
    #help - 参数的帮助信息，当指定为 argparse.SUPPRESS 时表示不显示该参数的帮助信息.
    parser.add_argument('-gt', '--gd_path', default='test/20/mask', type=str, 
                        help='data of root path')

    parser.add_argument('-sur', '--sur_image', default=False, type=bool, 
                        help='data of root path') 
    
    config = parser.parse_args()
    return config






# pixel accuracy
def binary_pa(s, g):
    """
        calculate the pixel accuracy of two N-d volumes.
        s: the segmentation volume of numpy array
        g: the ground truth volume of numpy array
        """
    pa = ((s == g).sum()) / g.size
    return pa


# Dice evaluation
def binary_dice(s, g):
    """
    calculate the Dice score of two N-d volumes.
    s: the segmentation volume of numpy array
    g: the ground truth volume of numpy array
    """
    assert (len(s.shape) == len(g.shape))
    # prod = np.multiply(s, g)
    # s0 = prod.sum()
    # dice = (2.0 * s0 + 1e-10) / (s.sum() + g.sum() + 1e-10)
    dice = metric.binary.dc(s, g)
    return dice



# IOU evaluation
def binary_iou(s, g):
    assert (len(s.shape) == len(g.shape))
    # 两者相乘值为1的部分为交集
    intersecion = np.multiply(s, g)
    # 两者相加，值大于0的部分为交集
    union = np.asarray(s + g > 0, np.float32)
    iou = intersecion.sum() / (union.sum() + 1e-10)
    return iou


# Hausdorff and ASSD evaluation
def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if (dim == 2):
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
    ero = ndimage.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge


def binary_hausdorff95(s, g, spacing=None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim == len(g.shape))
    if (spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim == len(spacing))
    img = np.zeros_like(s)
    if (image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim == 3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    dist1 = dist_list1[int(len(dist_list1) * 0.95)]
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    dist2 = dist_list2[int(len(dist_list2) * 0.95)]
    return max(dist1, dist2)


# 平均表面距离
def binary_assd(s, g, spacing=None):
    """
    get the average symetric surface distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim == len(g.shape))
    if (spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim == len(spacing))
    img = np.zeros_like(s)
    if (image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim == 3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    ns = s_edge.sum()
    ng = g_edge.sum()
    s_dis_g_edge = s_dis * g_edge
    g_dis_s_edge = g_dis * s_edge
    assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng)
    return assd


# relative volume error evaluation
def binary_relative_volume_error(s_volume, g_volume):
    s_v = float(s_volume.sum())
    g_v = float(g_volume.sum())
    assert (g_v > 0)
    rve = abs(s_v - g_v) / g_v
    return rve


def compute_class_sens_spec(pred, label):
    """
    Compute sensitivity and specificity for a particular example
    for a given class for binary.
    Args:
        pred (np.array): binary arrary of predictions, shape is
                         (height, width, depth).
        label (np.array): binary array of labels, shape is
                          (height, width, depth).
    Returns:
        sensitivity (float): precision for given class_num.
        specificity (float): recall for given class_num
    """
    tp = np.sum((pred == 1) & (label == 1))
    tn = np.sum((pred == 0) & (label == 0))
    fp = np.sum((pred == 1) & (label == 0))
    fn = np.sum((pred == 0) & (label == 1))

    # sensitivity = tp / (tp + fn)
    sensitivity = metric.binary.sensitivity(pred, label)
    specificity = tn / (tn + fp)

    return sensitivity, specificity


def get_evaluation_score(s_volume, g_volume, spacing, metric):
    if (len(s_volume.shape) == 4):
        assert (s_volume.shape[0] == 1 and g_volume.shape[0] == 1)
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    if (s_volume.shape[0] == 1):
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    metric_lower = metric.lower()

    if (metric_lower == "dice"):
        score = binary_dice(s_volume, g_volume)

    elif (metric_lower == "iou"):
        score,jc = binary_iou(s_volume, g_volume)
        print(score,jc)

    elif (metric_lower == 'assd'):
        score = binary_assd(s_volume, g_volume, spacing)

    elif (metric_lower == "hausdorff95"):
        score = binary_hausdorff95(s_volume, g_volume, spacing)

    elif (metric_lower == "rve"):
        score = binary_relative_volume_error(s_volume, g_volume)

    elif (metric_lower == "volume"):
        voxel_size = 1.0
        for dim in range(len(spacing)):
            voxel_size = voxel_size * spacing[dim]
        score = g_volume.sum() * voxel_size
    else:
        raise ValueError("unsupported evaluation metric: {0:}".format(metric))

    return score


if __name__ == '__main__':


    # vars（）函数返回对象object的属性和属性值的字典对象
    config = vars(parse_args())


    #计算dice
    seg_path = config['seg_path']
    # seg_path = 'result/sur/40/323/18/0'
    
    #GT
    # gd_path = 'DATASET/nnUNet_raw/Dataset132_ibsr18/labelsTr'
    gd_path = config['gd_path']

    #保存
    # save_dir = 'output/sur_18/1/add_source_RealESRGAN_18_3'
    save_dir = seg_path
    seg = sorted(os.listdir(seg_path))

    dices = []
    hds = []
    rves = []
    case_name = []
    senss = []
    specs = []
    delete=[[], []]

    for name in seg:
        if not name.startswith('.') and name.endswith('nii.gz'):
            # 加载label and segmentation image
            seg_ = nib.load(os.path.join(seg_path, name))
            seg_arr = seg_.get_fdata().astype('float32')
            gd_ = nib.load(os.path.join(gd_path, name))
            gd_arr = gd_.get_fdata().astype('float32')
            case_name.append(name)


            # for a in range(seg_arr.shape[1]):
            #     if a%4 != 0:
            #         delete[0].append(a)
            # seg_arr = np.delete(seg_arr, delete[0], axis=1)
            delete[0] = []

            gd_arr = np.squeeze(gd_arr)
            seg_arr = np.squeeze(seg_arr)
            # 求hausdorff95距离
            hd_score = get_evaluation_score(seg_arr, gd_arr, spacing=None, metric='hausdorff95')
            hds.append(hd_score)

            # 求体积相关误差
            rve = get_evaluation_score(seg_arr, gd_arr, spacing=None, metric='rve')
            rves.append(rve)

            # 求dice
            dice = get_evaluation_score(seg_arr, gd_arr, spacing=None, metric='dice')
            dices.append(dice)
            print(name,dice)

            
            # 敏感度，特异性
            sens, spec = compute_class_sens_spec(seg_arr, gd_arr)

            senss.append(sens)
            specs.append(spec)
    # 存入pandas
    case_name.append('avg')
    hds.append(np.mean(hds))
    rves.append(np.mean(rve))
    dices.append(np.mean(dices))
    senss.append(np.mean(senss))
    specs.append(np.mean(specs))

    # data = {'name':case_name, 'dice': dices, 'RVE': rves, 'Sens': senss, 'Spec': specs, 'HD95': hds}
    # df = pd.DataFrame(data=data, columns=['name', 'dice', 'RVE', 'Sens', 'Spec', 'HD95'])
    # df.to_csv(os.path.join(save_dir, 'metrics.csv'))

    data = {'dice': dices, 'RVE': rves, 'Sens': senss, 'Spec': specs, 'HD95': hds}
    df = pd.DataFrame(data=data, columns=['name', 'dice', 'RVE', 'Sens', 'Spec', 'HD95'], index=case_name)
    df.to_csv(os.path.join(save_dir, 'metrics_ct.csv'))
    print(config['seg_path'] + " ok")
