#python gen_data_stage1.py --input_path ../downstream_task/PraNet/revisedata/test/image/ --out_path ./REFUGE/test
#python gen_data_stage1.py --input_path ./ori/train_reject --out_path ./EyeQ/train/resized_reject
#python gen_data_stage1.py --input_path ./ori/test_reject --out_path ./EyeQ/test/resized_reject

python gen_data_stage1.py --input_path ../downstream_task/PraNet/revisedata/test/image/ --out_path ./REFUGE/test/resized_good/ --dsize 256
