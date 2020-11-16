# Rumor And Stance Detection
This project is a Multi-task Learning of rumor and stance detection. Using Pytorch library for the multi-task RNN model and FastText for word embedding.  
* The project was highly inspired by [*this*](https://core.ac.uk/download/pdf/286034844.pdf) paper.

## The Multi-task Model
Our model consists of 3 GRU layers:
- Task specific layer for Stance detection task
- Shared layer for both tasks
- Task specific layer for Rumor detection task  

<img src="https://github.com/Lzvitali/Rumor-And-Stance-Detection/blob/master/model/Multi-task%20model.PNG" alt="Multi-task model" width="520"/>

The inputs are vectors representing tweets after embedding with [*fastText*](https://fasttext.cc/) library.  

## Dataset  
The dataset is a combination of the below datasets:
-  [*RumourEval 2019 data*](https://figshare.com/articles/RumourEval_2019_data/8845580)
-  [*Twitter 15-16*](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?file_subpath=%2Frumor_detection_acl2017)

## Requirement
- Python >= 3.6    
- PyTorch 1.6.0  (there is a support for GPU and for CPU (no GPU))
- fasttext 0.9.2  

**For more information, explanations and outputs check out the [*notebook*](https://github.com/Lzvitali/Rumor-And-Stance-Detection/blob/master/Rumor%20And%20Stance%20Detection.ipynb)**
