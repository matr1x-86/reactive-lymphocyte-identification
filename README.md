# **Adversarial Training Collaborating Hybrid Convolution-Transformer Network for Automatic Identification of Reactive Lymphocytes in Peripheral Blood**

## *Introduction*
Reactive lymphocytes may indicate diseases, such as viral infection. The identification of these abnormal lymphocytes is vital to the diagnosis of disease and the assessment of disease severity. In recent years, pathologists have relied primarily on morphological methods to identify reactive lymphocytes, but this requires a great deal of effort and time. Additionally, cytological methods are subject to other challenges in their application, including high-resolution microscopy requirements, staining effects, and the presence of some similarities among different types of cells. We have introduced innovations in reactive lymphocyte recognition as well as improved accuracy, scalability, reduced bias, and real-time diagnostics. Our method has the advantage of combining convolution and Transformer, allowing the model to extract both local and global information within cells accurately. First, we collected peripheral blood samples, prepared smears and stained them, and then scanned the smears into images using an electron microscope. Next, to create the dataset, we used a cell detection framework to detect and crop cells in the images. As reactive lymphocytes may sometimes be confused with other peripheral blood cells, the dataset also includes eosinophils, neutrophils, lymphocytes, monocytes, and blasts. Finally, we input cell images into the model for training, allowing the model to learn the characteristics of various types of cells. In addition, we also enhanced the generalization of the model through virtual adversarial training. According to the results, on the validation set, the model achieved an accuracy of 93.80%, while on the test set, it achieved 93.6%.

## **Dataset**
We use peripheral blood cell images provided by the Laboratory of Hematology Zhongnan Hospital of Wuhan University as the dataset. There are six types of cells in the dataset, including eosinophil, lymphocyte, monocyte, neutrophil, blast and reactive lymphocyte.
```
/PeripheralBloodDataset/
  train/
    eosinophil/
      e1.jpg
    lymphocytes/
      l1.jpg
    monocyte/
      m1.jpg
    neutrophil/
      n1.jpg
    blast/
      nbl1.jpg
    reactive lymphocyte/
      vl.jpg
  val/
    ...
  test
    ...
```

## **Model**
We use [Next-ViT](https://github.com/bytedance/Next-ViT) as our identification model. Next-ViT is a model that combines the advantages of convolution and Transformer. Convolution helps the model extract detailed information in cells, while Transformer helps the model extract global information in cells.

<div style="text-align: center">
<img src="images/model.jpg" title="model" height="80%" width="80%">
</div>


## **Environment**
- OS: Ubuntu 23.04
- Software: Python 3.9.16
- GPU: NVIDIA RTX 3070 Ti

## **Installation**
First, clone the repository locally:
```
git clone https://github.com/matr1x-86/reactive-lymphocyte-identification.git
```

Then, install torch==1.10.0:
```
# CUDA 10.2
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch

# CUDA 11.3
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

Last, install remaining packages:
```
pip install -r requirements.txt
```

## **Training**
Get pertrained model from [here](https://drive.google.com/file/d/1b7ChnlT3uG3pTaZjtwYtnAaxAESF0MqK/view?usp=sharing).
```
cd reactive-lymphocyte-identification

bash train.sh --model nextvit_small --batch-size 32 --lr 3e-4 --warmup-epochs 0 --weight-decay 1e-8 --epochs 100 --sched step --decay-epochs 80 --input-size 224 --resume ../checkpoints/nextvit_small_in1k_224.pth --finetune --data-path your_imagenet_path
```
