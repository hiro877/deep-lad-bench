python lstm_demo.py --label_type next_log --feature_type sequentials --topk 10 --dataset HDFS --data_dir ../data/processed/HDFS/hdfs_0.0_tar
python lstm_demo.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/bgl_0.0_tar
python lstm_demo.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/Random/bgl_0.0_tar
python lstm_demo.py --label_type next_log --feature_type sequentials --topk 50 --dataset BGL --data_dir ../data/processed/BGL/Session/600/bgl_0.0_tar