#python transformer_demo.py --label_type next_log --feature_type sequentials --topk 10 --dataset HDFS --data_dir ../data/processed/HDFS/hdfs_0.0_tar
#python transformer_demo.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/bgl_0.0_tar --epoches 100
#python transformer_demo.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/Session/600/bgl_0.0_tar --epoches 100



#python transformer_demo_crossval.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --epoches 100 --data_dir ../data/processed/BGL/CrossVal/tr90a/bgl_1.0_tar --wtp_records_path wtp_records_transformer/val_a --epoches 10 --is_validation
#python transformer_demo_crossval.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --epoches 100 --data_dir ../data/processed/BGL/CrossVal/tr90a/bgl_1.0_tar --wtp_records_path wtp_records_transformer/val_b --epoches 10 --is_validation
#python transformer_demo_crossval.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --epoches 100 --data_dir ../data/processed/BGL/CrossVal/tr90a/bgl_1.0_tar --wtp_records_path wtp_records_transformer/val_c --epoches 10 --is_validation
#python transformer_demo_crossval.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --epoches 100 --data_dir ../data/processed/BGL/CrossVal/tr90a/bgl_1.0_tar --wtp_records_path wtp_records_transformer/val_d --epoches 10 --is_validation
#python transformer_demo_crossval.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --epoches 100 --data_dir ../data/processed/BGL/CrossVal/tr90a/bgl_1.0_tar --wtp_records_path wtp_records_transformer/val_e --epoches 10 --is_validation

# IHITE2023
#python transformer_demo_crossval.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --epoches 100 --data_dir ../data/processed/BGL/CrossVal/tr90a/bgl_1.0_tar --wtp_records_path IHIET2023/wtp_records_transformer/a --epoches 10
#python transformer_demo_crossval.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --epoches 100 --data_dir ../data/processed/BGL/CrossVal/tr90a_unsupervised/bgl_1.0_tar --wtp_records_path IHIET2023/wtp_records_transformer/unsupervised_a --epoches 10

# IntechOpen
#python transformer_demo_crossval.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --epoches 100 --data_dir ../data/processed/BGL/CrossVal/unsupervised/tr90c/bgl_1.0_tar/ --wtp_records_path Intechopen/wtp_records_transformer/shuffle_tr90c_use_attention --epoches 10 --use_attention

python transformer_demo_crossval.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --epoches 100 --data_dir ../data/processed/BGL/CrossVal/unsupervised/tr90a/bgl_1.0_tar/ --wtp_records_path Intechopen/wtp_records_transformer/val_shuffle_tr90a_use_attention --epoches 10 --use_attention --is_validation
#python transformer_demo_crossval.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --epoches 100 --data_dir ../data/processed/BGL/CrossVal/unsupervised/tr90b/bgl_1.0_tar/ --wtp_records_path Intechopen/wtp_records_transformer/val_shuffle_tr90b_use_attention --epoches 10 --use_attention --is_validation
python transformer_demo_crossval.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --epoches 100 --data_dir ../data/processed/BGL/CrossVal/unsupervised/tr90c/bgl_1.0_tar/ --wtp_records_path Intechopen/wtp_records_transformer/val_shuffle_tr90c_use_attention --epoches 10 --use_attention --is_validation
python transformer_demo_crossval.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --epoches 100 --data_dir ../data/processed/BGL/CrossVal/unsupervised/tr90d/bgl_1.0_tar/ --wtp_records_path Intechopen/wtp_records_transformer/val_shuffle_tr90d_use_attention --epoches 10 --use_attention --is_validation
python transformer_demo_crossval.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --epoches 100 --data_dir ../data/processed/BGL/CrossVal/unsupervised/tr90e/bgl_1.0_tar/ --wtp_records_path Intechopen/wtp_records_transformer/val_shuffle_tr90e_use_attention --epoches 10 --use_attention --is_validation