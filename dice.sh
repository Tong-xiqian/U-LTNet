for fold in {0..4}
do 

    # python dice.py -seg result/dell/40/18/2d/$fold  -gt test/dell/18/mask 
    # python dice.py -seg result/dell/40/18/3d/$fold  -gt test/dell/18/mask

    # python dice.py -seg result/dell/oasis/18/2d/$fold  -gt test/dell/18/mask
    # python dice.py -seg result/dell/oasis/18/3d/$fold  -gt test/dell/18/mask


    # python dice.py -seg result/dell/18/40/2d/$fold  -gt test/dell/40/mask
    # python dice.py -seg result/dell/18/40/3d/$fold  -gt test/dell/40/mask

    # python dice.py -seg result/dell/oasis/40/2d/$fold  -gt test/dell/40/mask
    # python dice.py -seg result/dell/oasis/40/3d/$fold  -gt test/dell/40/mask


    # python dice.py -seg result/dell/18/oasis/2d/$fold  -gt test/dell/oasis/mask   
    # python dice.py -seg result/dell/18/oasis/3d/$fold  -gt test/dell/oasis/mask   

    # python dice.py -seg result/dell/40/oasis/2d/$fold  -gt test/dell/oasis/mask   
    # python dice.py -seg result/dell/40/oasis/3d/$fold  -gt test/dell/oasis/mask  


    # python dice.py -seg result/20_20/2d/$fold  -gt test/20/mask 
    # python dice.py -seg result/20_20/3d/$fold  -gt test/20/mask 

    # python dice.py -seg result/18/20/2d/$fold  -gt test/20/mask 
    # python dice.py -seg result/18/20/3d/$fold  -gt test/20/mask 

    # python dice.py -seg result/40/20/2d/$fold  -gt test/20/mask 
    # python dice.py -seg result/40/20/3d/$fold  -gt test/20/mask 

    # python dice.py -seg result/oasis/20/2d/$fold  -gt test/20/mask 
    # python dice.py -seg result/oasis/20/3d/$fold  -gt test/20/mask    


    # python dice.py -seg result/sur/18/20/ibsr18/$fold  -gt test/20/mask 
    # python dice.py -seg result/sur/18/20/lbpa40/$fold  -gt test/20/mask 
    # python dice.py -seg result/sur/18/20/oasis/$fold  -gt test/20/mask 


    # python dice.py -seg result/sur/18/40/$fold  -gt test/dell/40/mask 
    # python dice.py -seg result/sur/18/oasis/$fold  -gt test/dell/oasis/mask 

    

    # python dice.py -seg result/sur/40/20/ibsr18/$fold  -gt test/20/mask 
    # python dice.py -seg result/sur/40/20/lbpa40/$fold  -gt test/20/mask 
    # python dice.py -seg result/sur/40/20/oasis/$fold  -gt test/20/mask 

    # python dice.py -seg result/sur/40/18/$fold  -gt test/dell/18/mask 
    # python dice.py -seg result/sur/40/oasis/$fold  -gt test/dell/oasis/mask 



    # python dice.py -seg result/sur/oasis/20/ibsr18/$fold  -gt test/20/mask 
    # python dice.py -seg result/sur/oasis/20/lbpa40/$fold  -gt test/20/mask 
    # python dice.py -seg result/sur/oasis/20/oasis/$fold  -gt test/20/mask 

    # python dice.py -seg result/sur/oasis/40/$fold  -gt test/dell/40/mask 
    # python dice.py -seg result/sur/oasis/18/$fold  -gt test/dell/18/mask


    # python dice.py -seg result/sur/oasis/40/$fold  -gt test/dell/40/mask 
    # python dice.py -seg result/sur/oasis/18/$fold  -gt test/dell/18/mask



    python dice.py -seg result/sur/40/322/18/$fold  -gt DATASET/nnUNet_raw/Dataset132_ibsr18/labelsTr
    python dice.py -seg result/sur/oasis/331/$fold  -gt DATASET/nnUNet_raw/Dataset132_ibsr18/labelsTr


    python dice.py -seg result/sur/40/322/oasis/$fold  -gt DATASET/nnUNet_raw/Dataset131_oasa/labelsTr
    python dice.py -seg result/sur/18/323/oasis/$fold  -gt DATASET/nnUNet_raw/Dataset131_oasa/labelsTr


done

# source dice.sh
python dice.py -seg result_gai/sur/o_18/0  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset132_ibsr18/labelsTr
python dice.py -seg result_gai/sur/o_18/1  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset132_ibsr18/labelsTr
python dice.py -seg result_gai/sur/o_18/2  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset132_ibsr18/labelsTr
python dice.py -seg result_gai/sur/o_18/3  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset132_ibsr18/labelsTr
python dice.py -seg result_gai/sur/o_18/4  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset132_ibsr18/labelsTr
1


python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset127_BraTS2021/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset127_BraTS2021/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset127_BraTS2021/nnUNetTrainer__nnUNetPlans__2d/fold_1/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset127_BraTS2021/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset127_BraTS2021/nnUNetTrainer__nnUNetPlans__2d/fold_2/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset127_BraTS2021/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset127_BraTS2021/nnUNetTrainer__nnUNetPlans__2d/fold_3/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset127_BraTS2021/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset127_BraTS2021/nnUNetTrainer__nnUNetPlans__2d/fold_4/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset127_BraTS2021/labelsTr
1

python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset127_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset127_BraTS2021/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset127_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset127_BraTS2021/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset127_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset127_BraTS2021/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset127_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset127_BraTS2021/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset127_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset127_BraTS2021/labelsTr
1

python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset122_MRILBPA40/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset122_MRILBPA40/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset122_MRILBPA40/nnUNetTrainer__nnUNetPlans__2d/fold_1/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset122_MRILBPA40/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset122_MRILBPA40/nnUNetTrainer__nnUNetPlans__2d/fold_2/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset122_MRILBPA40/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset122_MRILBPA40/nnUNetTrainer__nnUNetPlans__2d/fold_3/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset122_MRILBPA40/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset122_MRILBPA40/nnUNetTrainer__nnUNetPlans__2d/fold_4/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset122_MRILBPA40/labelsTr
1

python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset122_MRILBPA40/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset122_MRILBPA40/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset122_MRILBPA40/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset122_MRILBPA40/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset122_MRILBPA40/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset122_MRILBPA40/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset122_MRILBPA40/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset122_MRILBPA40/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset122_MRILBPA40/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset122_MRILBPA40/labelsTr
1

python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset423_MRIIBSR18/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset123_MRIIBSR18/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset423_MRIIBSR18/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset123_MRIIBSR18/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset423_MRIIBSR18/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset123_MRIIBSR18/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset423_MRIIBSR18/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset123_MRIIBSR18/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset423_MRIIBSR18/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset123_MRIIBSR18/labelsTr
1

python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset127_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset127_BraTS2021/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset431_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/deconder/validation -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset127_BraTS2021/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset431_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/encoder/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset127_BraTS2021/labelsTr
python dice.py -seg C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_trained_models/Dataset431_BraTS2021/all/fold_3/validation  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset127_BraTS2021/labelsTr
1


python dice.py -seg result_gai/sur/oasis/0  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset131_oasa/labelsTr
python dice.py -seg result_gai/sur/oasis/1  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset131_oasa/labelsTr
python dice.py -seg result_gai/sur/oasis/2  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset131_oasa/labelsTr
python dice.py -seg result_gai/sur/oasis/3  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset131_oasa/labelsTr
python dice.py -seg result_gai/sur/oasis/4  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset131_oasa/labelsTr
1

python dice.py -seg result_gai/sur/o_18/0  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset123_MRIIBSR18/labelsTr
python dice.py -seg result_gai/sur/o_18/1  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset123_MRIIBSR18/labelsTr
python dice.py -seg result_gai/sur/o_18/2  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset123_MRIIBSR18/labelsTr
python dice.py -seg result_gai/sur/o_18/3  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset123_MRIIBSR18/labelsTr
python dice.py -seg result_gai/sur/o_18/4  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset123_MRIIBSR18/labelsTr
1
python dice.py -seg result_gai/sur/18_o/0  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset131_oasa/labelsTr
python dice.py -seg result_gai/sur/18_o/1  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset131_oasa/labelsTr
python dice.py -seg result_gai/sur/18_o/2  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset131_oasa/labelsTr
python dice.py -seg result_gai/sur/18_o/3  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset131_oasa/labelsTr
python dice.py -seg result_gai/sur/18_o/4  -gt C:/Users/Admin/Desktop/train/DATA/nnunet/DATASET/nnUNet_raw/Dataset131_oasa/labelsTr
1