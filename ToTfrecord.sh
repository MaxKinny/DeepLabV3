export PYTHONPATH=$PYTHONPATH:`pwd`/models-master/research:`pwd`/models-master/research/slim
python ./models-master/research/deeplab/datasets/build_voc2012_data.py --image_folder=./Data/Database/JPEGImages --semantic_segmentation_folder=./Data/Database/SegmentationClass --list_folder=./Data/Database/ImageSets/Segmentation --output_dir=./Data/tfrecord


