for fold in {0..4}
do 
    # echo "nnUNetv2_train 1 3d_lowres $fold"

    nnUNetv2_train 431 2d $fold 

    # nnUNetv2_train 122 3d_cascade_fullres $fold

    c

    # nnUNetv2_predict -i /media/x/1e64ee43-5b31-404d-8aa8-65894df7f2a9/nnUNet-master/data/Dataset120_MRI/imagesTs -o output/18/2d/$fold -d 120 -c 2d -f $fold 

    # nnUNetv2_predict -i test/20/img -o result/40/20/2d/$fold -d 122 -c 2d -f $fold 

    # nnUNetv2_predict -i test/oasa/img -o result/40/oasis/3d/$fold -d 122 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/oasa/img -o result/40/oasis/2d/$fold -d 122 -c 2d -f $fold 

    # nnUNetv2_predict -i test/oasa/img -o result/18/oasis/3d/$fold -d 132 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/oasa/img -o result/18/oasis/2d/$fold -d 132 -c 2d -f $fold 

    # nnUNetv2_predict -i test/40/img -o result/18/40/3d/$fold -d 132 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/40/img -o result/18/40/2d/$fold -d 132 -c 2d -f $fold 

    # nnUNetv2_predict -i test/20/img -o result/18/20/3d/$fold -d 132 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/20/img -o result/18/20/2d/$fold -d 132 -c 2d -f $fold 

    # nnUNetv2_predict -i test/18/img -o result/oasis/18/3d/$fold -d 131 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/18/img -o result/oasis/18/2d/$fold -d 131 -c 2d -f $fold 

    # nnUNetv2_predict -i test/40/img -o result/oasis/40/3d/$fold -d 131 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/40/img -o result/oasis/40/2d/$fold -d 131 -c 2d -f $fold 



    # nnUNetv2_predict -i test/dell/18/img -o result/dell/oasis/18/3d/$fold -d 131 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/dell/18/img -o result/dell/oasis/18/2d/$fold -d 131 -c 2d -f $fold 

    # nnUNetv2_predict -i test/dell/40/img -o result/dell/oasis/40/3d/$fold -d 131 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/dell/40/img -o result/dell/oasis/40/2d/$fold -d 131 -c 2d -f $fold 


    # nnUNetv2_predict -i test/dell/40/img -o result/dell/18/40/3d/$fold -d 132 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/dell/40/img -o result/dell/18/40/2d/$fold -d 132 -c 2d -f $fold 

    # nnUNetv2_predict -i test/dell/oasis/img -o result/dell/18/oasis/3d/$fold -d 132 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/dell/oasis/img -o result/dell/18/oasis/2d/$fold -d 132 -c 2d -f $fold 



    # nnUNetv2_predict -i test/dell/18/img -o result/dell/40/18/3d/$fold -d 122 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/dell/18/img -o result/dell/40/18/2d/$fold -d 122 -c 2d -f $fold  

    # nnUNetv2_predict -i test/dell/oasis/img -o result/dell/40/oasis/3d/$fold -d 122 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/dell/oasis/img -o result/dell/40/oasis/2d/$fold -d 122 -c 2d -f $fold  

    # nnUNetv2_predict -i DATASET/nnUNet_raw/Dataset133_ibsr20/imagesTs -o result/20_20/3d/$fold -d 133 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i DATASET/nnUNet_raw/Dataset133_ibsr20/imagesTs -o result/20_20/2d/$fold -d 133 -c 2d -f $fold  
    

    # nnUNetv2_predict -i test/20/img -o result/oasis/20/3d/$fold -d 131 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/20/img -o result/oasis/20/2d/$fold -d 131 -c 2d -f $fold  





    # nnUNetv2_predict -i test/sur/18 -o result/sur/40/18/$fold -d 122 -c 3d_fullres -f $fold 

    # nnUNetv2_predict -i test/sur/20/ibsr18 -o result/sur/40/20/ibsr18/$fold -d 122 -c 3d_fullres -f $fold
    # nnUNetv2_predict -i test/sur/20/lbpa40 -o result/sur/40/20/lbpa40/$fold -d 122 -c 3d_fullres -f $fold
    # nnUNetv2_predict -i test/sur/20/oasis -o result/sur/40/20/oasis/$fold -d 122 -c 3d_fullres -f $fold

    # nnUNetv2_predict -i test/sur/oasis -o result/sur/40/oasis/$fold -d 122 -c 3d_fullres -f $fold


    # nnUNetv2_predict -i test/sur/20/ibsr18 -o result/sur/18/20/ibsr18/$fold -d 123 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/sur/20/lbpa40 -o result/sur/18/20/lbpa40/$fold -d 123 -c 3d_fullres -f $fold
    # nnUNetv2_predict -i test/sur/20/oasis -o result/sur/18/20/oasis/$fold -d 123 -c 3d_fullres -f $fold

    # nnUNetv2_predict -i test/sur/40 -o result/sur/18/40$fold -d 123 -c 3d_fullres -f $fold
    # nnUNetv2_predict -i test/sur/oasis -o result/sur/18/oasis/$fold -d 123 -c 3d_fullres -f $fold


    # nnUNetv2_predict -i test/sur/40 -o result/sur/oasis/40/$fold -d 131 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/sur/18 -o result/sur/oasis/18/$fold -d 231 -c 3d_fullres -f $fold 

    # nnUNetv2_predict -i test/sur/20/ibsr18 -o result/sur/oasis/ibsr18/20/$fold -d 131 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/sur/20/lbpa40  -o result/sur/oasis/20/lbpa40/$fold -d 131 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/sur/20/oasis -o result/sur/oasis/20/oasis/$fold -d 131 -c 3d_fullres -f $fold 

    # nnUNetv2_train 322 2d $fold
    # nnUNetv2_train 323 2d $fold




    # nnUNetv2_predict -i test/sur/18 -o result/sur/40/322/18/$fold -d 322 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/sur/18 -o result/sur/40/322/18/$fold -d 322 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/sur/18 -o result/sur/40/322/18/$fold -d 322 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/sur/18 -o result/sur/40/322/18/$fold -d 322 -c 3d_fullres -f $fold 

    # nnUNetv2_predict -i test/sur/oasis -o result/sur/40/322/oasis/$fold -d 322 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/sur/oasis -o result/sur/40/322/oasis/$fold -d 322 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/sur/oasis -o result/sur/40/322/oasis/$fold -d 322 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/sur/oasis -o result/sur/40/322/oasis/$fold -d 322 -c 3d_fullres -f $fold 

    # nnUNetv2_predict -i test/sur/oasis -o result/sur/18/323/oasis/$fold -d 323 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/sur/oasis -o result/sur/18/323/oasis/$fold -d 323 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/sur/oasis -o result/sur/18/323/oasis/$fold -d 323 -c 3d_fullres -f $fold 
    # nnUNetv2_predict -i test/sur/oasis -o result/sur/18/323/oasis/$fold -d 323 -c 3d_fullres -f $fold 

    # nnUNetv2_train 331 2d 3
    # sleep 10m  
    # nnUNetv2_train 331 2d 4
    # sleep 10m  

    # nnUNetv2_train 331 3d_fullres 2
    # sleep 10m  

    # nnUNetv2_train 331 3d_fullres 3
    # sleep 10m  

    # nnUNetv2_train 331 3d_fullres 4
    # sleep 10m  

    # nnUNetv2_predict -i test/sur/18 -o result/sur/oasis/331/$fold -d 431 -c 3d_fullres -f $fold

done

# for fold in {0..4}
# do 
#     nnUNetv2_train 331 2d $fold
#     nnUNetv2_train 331 3d_fullres $fold
# done 
# for fold in {0..4}
# do 
#     # echo "nnUNetv2_train 1 3d_lowres $fold"
#     nnUNetv2_train 122 3d_fullres $fold 
# done

# source tst.sh

# nnUNetv2_train 122 3d_lowres $fold

# nnUNetv2_train 122 3d_cascade_fullres $fold

# nnUNetv2_plan_and_preprocess -d 123 --verify_dataset_integrity

# nnUNetv2_predict -i ${nnUNet_raw}/Dataset131_WORD/ImagesTs -o output -d 131 -c 3d_fullres -f 1


# 使用说明
# 第一步：nnUNetv2_plan_and_preprocess -d 124 -npfp 12 --verify_dataset_integrity -gpu_memory_target 16 -np 12 进行数据预处理，种类3d_lowres，3d_cascade_fullres，2d，3d_lowres_fullres
# 第二步：nnUNetv2_train 124 2d 0     进行数据训练
# 第三步：nnUNetv2_predict -i intput -o output -d 126 -c 2d -f 0 //3d_fullres   进行验证

# nnUNetv2_predict -i data/18/1/RealESRGAN_gai_8_18_1_train -o output/sur_18/1/RealESRGAN_gai_8_18_1_train -d 122 -c 3d_fullres -f 0

# nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
# nnUNetv2_predict -h 查看更多参数解析
# INPUT_FOLDER: 测试数据地址
# OUTPUT_FOLDER： 分割数据存放地址
# CONFIGURATION： 使用的什么架构，2d or 3d_fullres or 3d_cascade_fullres
# save_probabilities：将预测概率与需要大量磁盘空间的预测分段掩码一起保存。
# npfp 特征提取进程数
# gpu_memory_target gpu显存
# np 进程数



# nnUNetv2_plan_and_preprocess -d 133 --verify_dataset_integrity
# nnUNetv2_predict -i test/20/img -o result/40/20/3d/5 -d 122 -c 3d_fullres -f 4

# nnUNetv2_train 124 3d_fullres 0 
# nnUNetv2_plan_and_preprocess -d 331 --verify_dataset_integrity

# nnUNetv2_predict -i test/sur/oasis -o aaaa -d 222 -c 3d_fullres -f 4

# nnUNetv2_predict -i test/sur/18 -o result/sur/40/323/18/0 -d 322 -c 3d_fullres -f 0


# nnUNetv2_predict -i test/sur/18 -o result/sur/oasis/331/4 -d 331 -c 3d_fullres -f 4


nnUNetv2_plan_and_preprocess -d 501 -npfp 12 --verify_dataset_integrity -gpu_memory_target 16 -np 12
nnUNetv2_plan_and_preprocess -d 502 -npfp 12 --verify_dataset_integrity -gpu_memory_target 16 -np 12
nnUNetv2_plan_and_preprocess -d 503 -npfp 12 --verify_dataset_integrity -gpu_memory_target 16 -np 12
nnUNetv2_plan_and_preprocess -d 504 -npfp 12 --verify_dataset_integrity -gpu_memory_target 16 -np 12

nnUNetv2_plan_and_preprocess -d 127 -npfp 12 --verify_dataset_integrity -gpu_memory_target 16 -np 12
nnUNetv2_plan_and_preprocess -d 122 -npfp 12 --verify_dataset_integrity -gpu_memory_target 16 -np 12
nnUNetv2_plan_and_preprocess -d 123 -npfp 12 --verify_dataset_integrity -gpu_memory_target 16 -np 12
nnUNetv2_plan_and_preprocess -d 422 -npfp 12 --verify_dataset_integrity -gpu_memory_target 9 -np 12
nnUNetv2_plan_and_preprocess -d 423 -npfp 12 --verify_dataset_integrity -gpu_memory_target 9 -np 12
nnUNetv2_plan_and_preprocess -d 431 -npfp 12 --verify_dataset_integrity -gpu_memory_target 8 -np 12

nnUNetv2_plan_and_preprocess -d 422 -npfp 12 --verify_dataset_integrity -gpu_memory_target 9 -np 12
nnUNetv2_plan_and_preprocess -d 423 -npfp 12 --verify_dataset_integrity -gpu_memory_target 7 -np 12
nnUNetv2_plan_and_preprocess -d 427 -npfp 12 --verify_dataset_integrity -gpu_memory_target 8 -np 12

nnUNetv2_plan_and_preprocess -d 127 -npfp 12 --verify_dataset_integrity -gpu_memory_target 8 -np 12




nnUNetv2_train 122 2d 1
nnUNetv2_train 123 2d 1


nnUNetv2_train 122 2d 2
nnUNetv2_train 123 2d 2


nnUNetv2_train 122 2d 3
nnUNetv2_train 123 2d 3


nnUNetv2_train 122 2d 4
nnUNetv2_train 123 2d 4
guokao




nnUNetv2_train 122 3d_fullres 0
nnUNetv2_train 123 3d_fullres 0

nnUNetv2_train 122 3d_fullres 1
nnUNetv2_train 123 3d_fullres 1


nnUNetv2_train 122 3d_fullres 2
nnUNetv2_train 123 3d_fullres 2


nnUNetv2_train 122 3d_fullres 3
nnUNetv2_train 123 3d_fullres 3


nnUNetv2_train 122 3d_fullres 4
nnUNetv2_train 123 3d_fullres 4




nnUNetv2_train 127 2d 0
nnUNetv2_train 127 2d 1
nnUNetv2_train 127 2d 2
nnUNetv2_train 127 2d 3
nnUNetv2_train 127 2d 4

nnUNetv2_train 127 3d_fullres 0
nnUNetv2_train 127 3d_fullres 1
nnUNetv2_train 127 3d_fullres 2
nnUNetv2_train 127 3d_fullres 3
nnUNetv2_train 127 3d_fullres 4
1

nnUNetv2_train 502 2d 0
nnUNetv2_train 502 2d 1
nnUNetv2_train 502 2d 2
nnUNetv2_train 502 2d 3
nnUNetv2_train 502 2d 4

nnUNetv2_train 502 3d_fullres 0
nnUNetv2_train 502 3d_fullres 1
nnUNetv2_train 502 3d_fullres 2
nnUNetv2_train 502 3d_fullres 3
nnUNetv2_train 502 3d_fullres 4

nnUNetv2_train 422 2d 0
nnUNetv2_train 422 2d 1
nnUNetv2_train 422 2d 2
nnUNetv2_train 422 2d 3
nnUNetv2_train 422 2d 4
1
nnUNetv2_train 422 3d_fullres 0
nnUNetv2_train 422 3d_fullres 1
nnUNetv2_train 422 3d_fullres 2
nnUNetv2_train 422 3d_fullres 3
nnUNetv2_train 422 3d_fullres 4
1
nnUNetv2_train 423 2d 0
nnUNetv2_train 423 2d 1
nnUNetv2_train 423 2d 2
nnUNetv2_train 423 2d 3
nnUNetv2_train 423 2d 4








nnUNetv2_train 423 3d_fullres 0
nnUNetv2_train 423 3d_fullres 1
nnUNetv2_train 423 3d_fullres 2
nnUNetv2_train 423 3d_fullres 3 
nnUNetv2_train 423 3d_fullres 4
1
nnUNetv2_train 427 3d_fullres 0
nnUNetv2_train 427 3d_fullres 1
nnUNetv2_train 427 3d_fullres 2
nnUNetv2_train 427 3d_fullres 3 
nnUNetv2_train 427 3d_fullres 4

nnUNetv2_train 431 3d_fullres 0
nnUNetv2_train 431 3d_fullres 1
nnUNetv2_train 431 3d_fullres 2
nnUNetv2_train 431 3d_fullres 3 
nnUNetv2_train 431 3d_fullres 4
1

nnUNetv2_train 422 3d_lowres 0
nnUNetv2_train 422 3d_lowres 1
nnUNetv2_train 422 3d_lowres 2

nnUNetv2_train 422 3d_lowres 3
nnUNetv2_train 422 3d_lowres 4
1


nnUNetv2_train 422 3d_fullres 0
nnUNetv2_train 423 3d_fullres 2



nnUNetv2_train 127 3d_fullres 3
nnUNetv2_train 431 3d_fullres 3
1






nnUNetv2_train 431 2d 0
nnUNetv2_train 431 2d 1
nnUNetv2_train 431 2d 2
nnUNetv2_train 431 2d 3
nnUNetv2_train 431 2d 4

nnUNetv2_train 431 3d_fullres 0
nnUNetv2_train 431 3d_fullres 1
nnUNetv2_train 431 3d_fullres 2
nnUNetv2_train 431 3d_fullres 3
nnUNetv2_train 431 3d_fullres 4
1



nnUNetv2_train 502 2d 0
nnUNetv2_train 502 2d 1
nnUNetv2_train 502 2d 2
nnUNetv2_train 502 2d 3
nnUNetv2_train 502 2d 4

nnUNetv2_train 502 3d_fullres 0
nnUNetv2_train 502 3d_fullres 1
nnUNetv2_train 502 3d_fullres 2
nnUNetv2_train 502 3d_fullres 3
nnUNetv2_train 502 3d_fullres 4
1 


nnUNetv2_predict -i test/oasis/img -o result_gai/img_soure/oasis/0 -d 122 -c 3d_fullres -f 0
nnUNetv2_predict -i test/oasis/img -o result_gai/img_soure/oasis/1 -d 122 -c 3d_fullres -f 1
nnUNetv2_predict -i test/oasis/img -o result_gai/img_soure/oasis/2 -d 122 -c 3d_fullres -f 2
nnUNetv2_predict -i test/oasis/img -o result_gai/img_soure/oasis/3 -d 122 -c 3d_fullres -f 3
nnUNetv2_predict -i test/oasis/img -o result_gai/img_soure/oasis/4 -d 122 -c 3d_fullres -f 4
1
nnUNetv2_predict -i test/sur/oasis -o result_gai/sur/oasis/0 -d 422 -c 3d_fullres -f 0
nnUNetv2_predict -i test/sur/oasis -o result_gai/sur/oasis/1 -d 422 -c 3d_fullres -f 1
nnUNetv2_predict -i test/sur/oasis -o result_gai/sur/oasis/2 -d 422 -c 3d_fullres -f 2
nnUNetv2_predict -i test/sur/oasis -o result_gai/sur/oasis/3 -d 422 -c 3d_fullres -f 3
nnUNetv2_predict -i test/sur/oasis -o result_gai/sur/oasis/4 -d 422 -c 3d_fullres -f 4
1

1


nnUNetv2_train 127 3d_fullres 2
nnUNetv2_train 127 3d_fullres 3
nnUNetv2_train 127 3d_fullres 4

nnUNetv2_train 431 3d_fullres 0
nnUNetv2_train 431 3d_fullres 2
nnUNetv2_train 431 3d_fullres 4
1
nnUNetv2_train 431 3d_fullres 3
nnUNetv2_train 431 3d_fullres 4
1
C:\Users\Admin\Desktop\train\DATA\nnunet\DATASET\nnUNet_raw\Dataset431_BraTS2021\imagesTs

nnUNetv2_predict -i C:\Users\Admin\Desktop\train\DATA\nnunet\DATASET\nnUNet_raw\Dataset431_BraTS2021\imagesTs -o lsunet\brtas/0 -d 431 -c 3d_fullres -f 0
nnUNetv2_predict -i C:\Users\Admin\Desktop\train\DATA\nnunet\DATASET\nnUNet_raw\Dataset431_BraTS2021\imagesTs -o lsunet\brtas/1 -d 431 -c 3d_fullres -f 1
nnUNetv2_predict -i C:\Users\Admin\Desktop\train\DATA\nnunet\DATASET\nnUNet_raw\Dataset431_BraTS2021\imagesTs -o lsunet\brtas/2 -d 431 -c 3d_fullres -f 2
nnUNetv2_predict -i C:\Users\Admin\Desktop\train\DATA\nnunet\DATASET\nnUNet_raw\Dataset431_BraTS2021\imagesTs -o lsunet\brtas/3 -d 431 -c 3d_fullres -f 3
nnUNetv2_predict -i C:\Users\Admin\Desktop\train\DATA\nnunet\DATASET\nnUNet_raw\Dataset431_BraTS2021\imagesTs -o result_gai/img_soure/oasis/4 -d 431 -c 3d_fullres -f 4
1
nnUNetv2_predict -i test/sur/18 -o result_gai/sur/18/0 -d 422 -c 3d_fullres -f 0
nnUNetv2_predict -i test/sur/18 -o result_gai/sur/18/1 -d 422 -c 3d_fullres -f 1
nnUNetv2_predict -i test/sur/18 -o result_gai/sur/18/2 -d 422 -c 3d_fullres -f 2
nnUNetv2_predict -i test/sur/18 -o result_gai/sur/18/3 -d 422 -c 3d_fullres -f 3
nnUNetv2_predict -i test/sur/18 -o result_gai/sur/18/4 -d 422 -c 3d_fullres -f 4

nnUNetv2_predict -i test/sur/18 -o result_gai/sur/o_18/0 -d 427 -c 3d_fullres -f 0
nnUNetv2_predict -i test/sur/18 -o result_gai/sur/o_18/1 -d 427 -c 3d_fullres -f 1
nnUNetv2_predict -i test/sur/18 -o result_gai/sur/o_18/2 -d 427 -c 3d_fullres -f 2
nnUNetv2_predict -i test/sur/18 -o result_gai/sur/o_18/3 -d 427 -c 3d_fullres -f 3
nnUNetv2_predict -i test/sur/18 -o result_gai/sur/o_18/4 -d 427 -c 3d_fullres -f 4
1

nnUNetv2_predict -i test/sur/oasis -o result_gai/sur/18_o/0 -d 423 -c 3d_fullres -f 0
nnUNetv2_predict -i test/sur/oasis -o result_gai/sur/18_o/1 -d 423 -c 3d_fullres -f 1
nnUNetv2_predict -i test/sur/oasis -o result_gai/sur/18_o/2 -d 423 -c 3d_fullres -f 2
nnUNetv2_predict -i test/sur/oasis -o result_gai/sur/18_o/3 -d 423 -c 3d_fullres -f 3
nnUNetv2_predict -i test/sur/oasis -o result_gai/sur/18_o/4 -d 423 -c 3d_fullres -f 4
1