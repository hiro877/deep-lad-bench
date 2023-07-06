#python lstm_demo.py --label_type anomaly --feature_type sequentials --topk 10 --dataset HDFS --data_dir ../data/processed/HDFS/hdfs_1.0_tar
#python lstm_demo.py --label_type anomaly --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/bgl_1.0_tar
#python lstm_demo.py --label_type anomaly --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/Session/600/bgl_1.0_tar

#python lstm_demo_crossval.py --label_type anomaly --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90a/bgl_1.0_tar --epoches 10 --is_validation
#python lstm_demo_crossval.py --label_type anomaly --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90a/bgl_1.0_tar --wtp_records_path wtp_records_lstm/val_c --epoches 10 --is_validation
#python lstm_demo_crossval.py --label_type anomaly --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90a/bgl_1.0_tar --wtp_records_path wtp_records_lstm/val_d --epoches 10 --is_validation
#python lstm_demo_crossval.py --label_type anomaly --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90a/bgl_1.0_tar --wtp_records_path wtp_records_lstm/val_e --epoches 10 --is_validation
#python lstm_demo_crossval.py --label_type anomaly --feature_type sequentials --use_attention --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90a/bgl_1.0_tar --wtp_records_path wtp_records_lstm/val_e --epoches 10 --is_validation


#===== IntechOpen =====
#python lstm_demo_crossval.py --label_type anomaly --feature_type sequentials --use_attention --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90a/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_lstm/shuffle_tr90a --epoches 10
#python lstm_demo_crossval.py --label_type anomaly --feature_type sequentials --use_attention --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90b/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_lstm/shuffle_tr90b --epoches 10
#python lstm_demo_crossval.py --label_type anomaly --feature_type sequentials --use_attention --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90c/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_lstm/shuffle_tr90c --epoches 10
#python lstm_demo_crossval.py --label_type anomaly --feature_type sequentials --use_attention --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90d/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_lstm/shuffle_tr90d --epoches 10
#python lstm_demo_crossval.py --label_type anomaly --feature_type sequentials --use_attention --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90e/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_lstm/shuffle_tr90e --epoches 10

#- Validation -
#python lstm_demo_crossval.py --label_type anomaly --feature_type sequentials --use_attention --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90a/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_lstm/val_shuffle_tr90a --epoches 10 --is_validation
#python lstm_demo_crossval.py --label_type anomaly --feature_type sequentials --use_attention --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90b/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_lstm/val_shuffle_tr90b --epoches 10 --is_validation
#python lstm_demo_crossval.py --label_type anomaly --feature_type sequentials --use_attention --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90c/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_lstm/val_shuffle_tr90c --epoches 10 --is_validation
#python lstm_demo_crossval.py --label_type anomaly --feature_type sequentials --use_attention --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90d/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_lstm/val_shuffle_tr90d --epoches 10 --is_validation
#python lstm_demo_crossval.py --label_type anomaly --feature_type sequentials --use_attention --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90e/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_lstm/val_shuffle_tr90e --epoches 10 --is_validation

python lstm_demo_crossval_no_duplicate.py --label_type anomaly --feature_type sequentials --use_attention --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90a/bgl_1.0_tar --wtp_records_path /work2/huchida/SaveLearnedFolder/deep-lad-bench/lstm/tr90a_val --epoches 10 --is_validation --dataset_name tr90a_val
python lstm_demo_crossval_no_duplicate.py --label_type anomaly --feature_type sequentials --use_attention --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90b/bgl_1.0_tar --wtp_records_path /work2/huchida/SaveLearnedFolder/deep-lad-bench/lstm/tr90b_val --epoches 10 --is_validation --dataset_name tr90b_val
python lstm_demo_crossval_no_duplicate.py --label_type anomaly --feature_type sequentials --use_attention --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90c/bgl_1.0_tar --wtp_records_path /work2/huchida/SaveLearnedFolder/deep-lad-bench/lstm/tr90c_val --epoches 10 --is_validation --dataset_name tr90c_val
python lstm_demo_crossval_no_duplicate.py --label_type anomaly --feature_type sequentials --use_attention --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90d/bgl_1.0_tar --wtp_records_path /work2/huchida/SaveLearnedFolder/deep-lad-bench/lstm/tr90d_val --epoches 10 --is_validation --dataset_name tr90d_val
python lstm_demo_crossval_no_duplicate.py --label_type anomaly --feature_type sequentials --use_attention --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90e/bgl_1.0_tar --wtp_records_path /work2/huchida/SaveLearnedFolder/deep-lad-bench/lstm/tr90e_val --epoches 10 --is_validation --dataset_name tr90e_val
