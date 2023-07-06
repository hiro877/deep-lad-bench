#python transformer_demo.py --label_type next_log --feature_type sequentials --topk 10 --dataset HDFS --data_dir ../data/processed/HDFS/hdfs_0.0_tar
python transformer_demo.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/bgl_0.0_tar --epoches 100
python transformer_demo.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/Session/600/bgl_0.0_tar --epoches 100


#python transformer_demo.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/bgl_0.0_tar --window_size 1
#python transformer_demo.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/bgl_0.0_tar --window_size 5
#python transformer_demo.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/bgl_0.0_tar --window_size 20
#python transformer_demo.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/bgl_0.0_tar --window_size 100
#python transformer_demo.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/bgl_0.0_tar --window_size 200


python transformer_demo.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --epoches 100 --data_dir ../data/processed/BGL/Criterion/tr78/bgl_1.0_tar --is_validation
python transformer_demo.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --epoches 100 --data_dir ../data/processed/BGL/Criterion/tr89/bgl_1.0_tar --epoches 10
python transformer_demo.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --epoches 10 --data_dir ../data/processed/BGL/Criterion/tr89/bgl_1.0_tar --is_validation
python transformer_demo_crossval.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90a/bgl_1.0_tar --wtp_records_path wtp_records_transformer/val_a --epoches 10

python transformer_demo.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --epoches 10 --data_dir ../data/processed/BGL/CrossVal/tr90a/bgl_1.0_tar --is_validation

python transformer_demo.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --epoches 10 --data_dir ../data/processed/BGL/CrossVal/criterion/bgl_1.0_tar --wtp_records_path wtp_records_transformer/val_criterion

--wtp_records_path wtp_records_transformer/val_c
