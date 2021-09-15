# Digital Health Hackathon - MLP baseline

## Setup

```shell
pyenv install 3.7.10
pyenv virtualenv 3.7.10 digital-health-hackathon
pyenv activate digital-health-hackathon
pip install -r requirements.txt
```

## Module 및 Package 설명 

### data_prep_utils 패키지

* get_data 함수: Data 폴더의 csv 파일들을 pandas DataFrame 으로 각각 읽어서 반환

#### data_loader

#### dataset 

#### preprocess


#### 

### config 모듈

전반적으로 사용할 불변 변수(epoch, input size, output size 등)를 담고 있는 모듈. 값을 overwrite하려 하면 오류를 발생시킴.


