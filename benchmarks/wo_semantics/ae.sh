#python ae_demo.py --feature_type sequentials --anomaly_ratio 0.03 --dataset HDFS --data_dir ../data/processed/HDFS/hdfs_0.0_tar
#python ae_demo.py --feature_type sequentials --anomaly_ratio 0.8 --dataset BGL --data_dir ../data/processed/BGL/bgl_0.0_tar
#python ae_demo_crossval.py --feature_type sequentials --anomaly_ratio 0.8 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90a/bgl_1.0_tar --wtp_records_path wtp_records_ae/val_a --epoches 10 --is_validation

# Validation
#python ae_demo_crossval.py --feature_type sequentials --anomaly_ratio 0.8 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/unsupervised/tr90a/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_ae/shuffle_tr90a --epoches 10 --is_validation
#python ae_demo_crossval.py --feature_type sequentials --anomaly_ratio 0.8 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/unsupervised/tr90b/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_ae/shuffle_tr90b --epoches 10 --is_validation
#python ae_demo_crossval.py --feature_type sequentials --anomaly_ratio 0.8 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/unsupervised/tr90c/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_ae/shuffle_tr90c --epoches 10 --is_validation
#python ae_demo_crossval.py --feature_type sequentials --anomaly_ratio 0.8 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/unsupervised/tr90d/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_ae/shuffle_tr90d --epoches 10 --is_validation
#python ae_demo_crossval.py --feature_type sequentials --anomaly_ratio 0.8 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/unsupervised/tr90e/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_ae/shuffle_tr90e --epoches 10 --is_validation

#python ae_demo_crossval.py --feature_type sequentials --anomaly_ratio 0.8 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/unsupervised/tr90a/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_ae/shuffle_tr90a --epoches 10
python ae_demo_crossval.py --feature_type sequentials --anomaly_ratio 0.8 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/unsupervised/tr90b/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_ae/shuffle_tr90b --epoches 10
python ae_demo_crossval.py --feature_type sequentials --anomaly_ratio 0.8 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/unsupervised/tr90c/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_ae/shuffle_tr90c --epoches 10
python ae_demo_crossval.py --feature_type sequentials --anomaly_ratio 0.8 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/unsupervised/tr90d/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_ae/shuffle_tr90d --epoches 10
python ae_demo_crossval.py --feature_type sequentials --anomaly_ratio 0.8 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/unsupervised/tr90e/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_ae/shuffle_tr90e --epoches 10
