环境：
```
ubuntu 16.04 + TensorFlow 1.9.0 + cuda 9.0 + cudnn 7.1.4 +python3.6。5
```
tensorflow 项目链接 
```
https://github.com/tensorflow
```
DeepLab位置：
```
https://github.com/tensorflow/models/tree/master/research/deeplab
```
1. **添加依赖库到PYTHONPATH**
git整个项目，然后在~/.bashrc中添加：
```
export PYTHONPATH=/home/public/Desktop/DeepLabV3+/models-master/research:/home/public/Desktop/DeepLabV3+/models-master/research/slim:$PYTHONPATH
```
2. **测试一下**
快速测试，调用model_test.py:
```
python /home/public/Desktop/DeepLabV3+/models-master/research/deeplab/model_test.py
```
若出现ok，就成功啦

3. **数据集转换为tfrecord格式**
运行：
```
python /home/public/Desktop/DeepLabV3+/models-master/research/deeplab/datasets/build_voc2012_data.py --image_folder=/home/public/Desktop/DeepLabV3+/Data/Database/JPEGImages --semantic_segmentation_folder=/home/public/Desktop/DeepLabV3+/Data/Database/SegmentationClass --list_folder=/home/public/Desktop/DeepLabV3+/Data/Database/ImageSets/Segmentation --output_dir=/home/public/Desktop/DeepLabV3+/Data/tfrecord
```

4. **注册数据集**
（1）更改文件：
```
/home/public/Desktop/DeepLabV3+/models-master/research/deeplab/dataset/segmentation_dataset.py
```
将
```
DATASETS_INFORMATION = { 
    'cityscapes': _CITYSCAPES_INFORMATION, 
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION, 
    'ade20k': _ADE20K_INFORMATION, 
}
```
改成：
```
DATASETS_INFORMATION = { 
    'cityscapes': _CITYSCAPES_INFORMATION, 
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION, 
    'ade20k': _ADE20K_INFORMATION, 
    'Database': _DATASET_NAME,   # 自己的数据集名字及对应配置放在这里，根据我的目录结构可见我的名字是Database
}
```
在上面一段加入：
```
_DATASET_NAME = DatasetDescriptor( 
    splits_to_sizes={ 
        'train': 10000,      # 这里根据Segmentation中的.txt文件名对应，
        'val': 2000,         # 这里根据Segmentation中的.txt文件名对应，
        'trainval': 12000    # 数字代表对应数据集包含图像的数量
    }, 
    num_classes=21,   # 类别数目。这里固定填写21，不然预测的时候会出错，原因不明。
    ignore_label=255,   # 有些数据集标注有白色描边（VOC 2012），不代表任何实际类别
)
```
（2）更改文件：
```
/home/public/Desktop/DeepLabV3+/models-master/research/deeplab/utils/train_utils.py
```
多加一个logits元素，作用是在使用预训练权重时候，不加载该层：
```
exclude_list = ['global_step','logits']
```
（3）更改文件：
```
/home/public/Desktop/DeepLabV3+/models-master/research/deeplab/common.py
```
将mobilenet_v2改为xception_65，如图：
![](leanote://file/getImage?fileId=5bcb1cb9ab64410bf5003d4b)
（4）更改文件：
```
/home/public/Desktop/DeepLabV3+/models-master/research/deeplab/trian.py
```
参数设置成：
```
initialize_last_layer = False   
last_layers_contain_logits_only = True
training_number_of_steps = 50000
train_crop_size = [321, 321]（由于内存不够，将其改小，但是crop_size至少要大于300，遵循的公式是(crop_size-1)/4为整数）
batch_size = 8
```
5. **下载预训练网络**
到[这里](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md)下载网络。我选择了 xception65_coco_voc_trainaug。下载解压后，将model.ckpt.data-00000-of-00001复制一份并重命名为ckpt格式，再将四个文件复制到weights文件夹（见我的目录结构）。
6. **开始训练**
运行：
```
python /home/public/Desktop/DeepLabV3+/models-master/research/deeplab/train.py --logtostderr --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=321 --train_crop_size=321 --train_batch_size=8 --training_number_of_steps=50000 --fine_tune_batch_norm=False --tf_initial_checkpoint=/home/public/Desktop/DeepLabV3+/weights/deeplabv3_pascal_train_aug/model.ckpt --train_logdir=/home/public/Desktop/DeepLabV3+/checkpoint --dataset_dir=/home/public/Desktop/DeepLabV3+/Data/tfrecord
```
7. **训练可视化**
运行：
```
~$:tensorboard /home/public/Desktop/DeepLabV3+/checkpoint
```
将输出的网址复制到浏览器即可。

8. **预测**
（1）更改文件：
```
/home/public/Desktop/DeepLabV3+/models-master/research/deeplab/utils/get_dataset_colormap.py
```
```
# Dataset names. 
_ADE20K = 'ade20k' 
_CITYSCAPES = 'cityscapes' 
_MAPILLARY_VISTAS = 'mapillary_vistas' 
_PASCAL = 'pascal' 
_DATASET_NAME='Database'   # 添加在这里，和注册的名字相同
```
```
# Max number of entries in the colormap for each dataset. 
_DATASET_MAX_ENTRIES = { 
    _ADE20K: 151, 
    _CITYSCAPES: 19, 
    _MAPILLARY_VISTAS: 66, 
    _PASCAL: 256, 
    _DATASET_NAME: 3,   # 在这里添加 colormap 的颜色数
}
```
接下来增加一个函数，用途是返回一个 np.ndarray 对象，尺寸为 [classes, 3] ，即colormap共有 classes 种RGB颜色，分别代表不同的类别。这个函数具体怎么写，还是由数据集的实际情况来定。
```
def create_dataset_name_label_colormap(): 
     return np.asarray([ 
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        ])
```
最后修改 create_label_colormap 函数，在这个调用接口中加上我们自己的数据集：
```
def create_label_colormap(dataset=_PASCAL): 
  """Creates a label colormap for the specified dataset. 

  Args: 
    dataset: The colormap used in the dataset. 

  Returns: 
    A numpy array of the dataset colormap. 

  Raises: 
    ValueError: If the dataset is not supported. 
  """ 
  if dataset == _ADE20K: 
    return create_ade20k_label_colormap() 
  elif dataset == _CITYSCAPES: 
    return create_cityscapes_label_colormap() 
  elif dataset == _MAPILLARY_VISTAS: 
    return create_mapillary_vistas_label_colormap() 
  elif dataset == _PASCAL: 
    return create_pascal_label_colormap() 
  elif dataset == _DATASET_NAME:             # 添加在这里
    return create_dataset_name_label_colormap()
  else:
    raise ValueError('Unsupported dataset.')
```
（2）更改文件：
```
/home/public/Desktop/DeepLabV3+/models-master/research/deeplab/vis.py
```
 - vis_split：设置为val
 - vis_crop_size:设置480,528为真实图片的大小
 - dataset：设置为我们在segmentation_dataset.py文件设置的数据集名称
 - dataset_dir：设置为创建的TFRecord
 - colormap_type：将第一个参数从pascal改为Database，再在第二个参数（数组）中添加一个新的元素Database
然后执行：
```
python /home/public/Desktop/DeepLabV3+/models-master/research/deeplab/vis.py --logtostderr --vis_split="val" -model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=480 --vis_crop_size=528 --checkpoint_dir='/home/public/Desktop/DeepLabV3+/checkpoint' --vis_logdir='/home/public/Desktop/DeepLabV3+/predict_result' --dataset_dir='/home/public/Desktop/DeepLabV3+/Data/tfrecord' 
```
9. **预测新的图片**
需要将图片复制到数据集中，然后重新执行第3步。预测的时候，vis.py读取的是tfrecord中的文件，database文件夹不参与其中。在执行第三步，生成tfrecord文件时，SegmentationClass文件夹中需要一个与JPEGImages文件夹同名的图片来欺骗程序使其继续运行下去。
