# Facial Expression Prediction


This repository is to do facial expression prediction by fine-tuning ResNet-101 with FER-2013 Faces Database.


## Dependencies

- Python 3.5.2
- PyTorch 1.0

## Dataset

I use the FER-2013 Faces Database, a set of 35,887 pictures of people displaying 7 emotional expressions (angry, disgusted, fearful, happy, sad, surprised and neutral).

 ![image](https://github.com/foamliu/Facial-Expression-Prediction/raw/master/images/random.png)

You can get it from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data), make sure fer2013.csv is in fer2013 folder.

## Usage

### Data Pre-processing
Extract 28,709 images [Usage='Training'] for training, and 3,589 [Usage='PublicTest'] for validation:
```bash
$ python pre-process.py
```
  
### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

 ![image](https://github.com/foamliu/Facial-Expression-Prediction-v2/raw/master/images/train.png)



### Analysis
Rename the best model to "Model.best.hdf5", put it in "models" folder, and use 3,589 images [Usage='PrivateTest'] for result analysis:
```bash
$ python analyze.py
```

#### Test acc: 
**65.46%**

#### Confusion matrix:

 ![image](https://github.com/foamliu/Facial-Expression-Prediction-v2/raw/master/images/confusion_matrix_not_normalized.png)

 ![image](https://github.com/foamliu/Facial-Expression-Prediction-v2/raw/master/images/confusion_matrix_normalized.png)


### Demo
Download [pre-trained model](https://github.com/foamliu/Facial-Expression-Prediction-v2/releases/download/v1.0/facial_expression.pt) then run:

```bash
$ python demo.py
```
