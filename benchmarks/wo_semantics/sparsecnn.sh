python sparcecnn_demo.py --dataset BGL --data_dir ../data/processed/BGL/bgl_1.0_tar --label_type anomaly --feature_type sequentials
python sparcecnn_demo.py --dataset BGL --data_dir ../data/processed/BGL/Criterion/bgl_1.0_tar --label_type anomaly --feature_type sequentials

python sparcecnn_demo.py --dataset BGL --label_type anomaly --feature_type sequentials --data_dir ../data/processed/BGL/MissLabeled/Random/0.01/bgl_1.0_tar
python sparcecnn_demo.py --dataset BGL --label_type anomaly --feature_type sequentials --data_dir ../data/processed/BGL/MissLabeled/Random/0.02/bgl_1.0_tar
python sparcecnn_demo.py --dataset BGL --label_type anomaly --feature_type sequentials --data_dir ../data/processed/BGL/MissLabeled/Random/0.03/bgl_1.0_tar
python sparcecnn_demo.py --dataset BGL --label_type anomaly --feature_type sequentials --data_dir ../data/processed/BGL/MissLabeled/Random/0.04/bgl_1.0_tar
python sparcecnn_demo.py --dataset BGL --label_type anomaly --feature_type sequentials --data_dir ../data/processed/BGL/MissLabeled/Random/0.05/bgl_1.0_tar
python sparcecnn_demo.py --dataset BGL --label_type anomaly --feature_type sequentials --data_dir ../data/processed/BGL/MissLabeled/Random/0.06/bgl_1.0_tar
python sparcecnn_demo.py --dataset BGL --label_type anomaly --feature_type sequentials --data_dir ../data/processed/BGL/MissLabeled/Random/0.07/bgl_1.0_tar
python sparcecnn_demo.py --dataset BGL --label_type anomaly --feature_type sequentials --data_dir ../data/processed/BGL/MissLabeled/Random/0.08/bgl_1.0_tar
python sparcecnn_demo.py --dataset BGL --label_type anomaly --feature_type sequentials --data_dir ../data/processed/BGL/MissLabeled/Random/0.09/bgl_1.0_tar
python sparcecnn_demo.py --dataset BGL --label_type anomaly --feature_type sequentials --data_dir ../data/processed/BGL/MissLabeled/Random/0.1/bgl_1.0_tar

python sparcecnn_demo_crossval.py --label_type anomaly --feature_type sequentials --dataset BGL --data_dir ../data/processed/BGL/CrossVal/tr90a/bgl_1.0_tar --wtp_records_path Intechopen/wtp_records_sparsecnn/val_shuffle_tr90a --epoches 10 --is_validation
