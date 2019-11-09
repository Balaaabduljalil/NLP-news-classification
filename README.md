# NLP - News classification

Train and deploy a news classifier based ULMFit.

- Live version: https://nlp.imadelhanafi.com
- Blog post: https://imadelhanafi.com/posts/text_classification_ulmfit/

<a href="https://nlp.imadelhanafi.com">
<img src="https://imadelhanafi.com/data/draft/nlp.png" width="500" height="300" class="center" />
</a>


# Running on cloud/local machine

To run the application, we can use the pre-build docker image available Docker hub and simply run the following command

```
docker run --rm -p 8080:8080 imadelh/news
```

The application will be available on http://0.0.0.0:8080

# Reproduce results

## LR and SVM

To reproduce results reported in the blog post, we need to have install the requirements in our development environment.

```
# Open requirement.txt and select torch==1.1.0 instead of the cpu version used for inference only.
# Then install requirement
pip install -r requirements.txt
```


After completing the installation, we can run parameters search or training of sk-learn models as follows

```
# Params search for SVM
cd sklearn_models
python3 params_search.py --model svc --exp_name svmsearch_all --data dataset_processed

# Params search for LR
python3 params_search.py --model lreg --exp_name logreg_all --data dataset_processed
```

The parameters space is defined in the file `sklearn_models/params_search.py`. The outputs of parameters search will be saved in the logs folder.

Training a model for a fixed set of parameters can be done using `sklearn_models/baseline.py`

```
# Specify the parameters of the model inside baseline.py and run
python3 baseline.py --model svc --exp_name svc_all --data dataset_processed
```

The logs/metrics on testing will be saved in `sklearn_models/logs/` and the trained model will be saved in `sklearn_models/saved_models/`


## ULMFit

To reproduce/train ULMFit model, the notebooks available in `ULMFIT` will be used. Same requirements are needed as explained before. We will need a GPU to train models, this can be done using Google Colab.
To be able to run the training we need to specify the path to a folder where the data is stored.

- Locally:

Save data from `data/`, then specify the absolute PATH in the beginning of the notebook.
```
# This is the absolute path to where folder "data" is available
PATH = "/app/analyse/"
```

- Google Colab:

Save the data in Google drive, for example `files/nlp/`

```
# The folder 'data' is saved in Google drive in "files/nlp/"

# While running the notebook from google colab, mount the drive and define PATH to data
from google.colab import drive
drive.mount('/content/gdrive/')

# then give the path where your data is stored (in google drive)
PATH = "/content/gdrive/My Drive/files/nlp/"
```

`01-ulmfit_balanced_dataset.ipynb` "<a href=\"https://colab.research.google.com/github/imadelh/NLP-news-classification/blob/master/ULMFIT/01-ulmfit_balanced_dataset.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>" - Train ULMfit on balanced dataset


`02_ulmfit_all_data.ipynb` "<a href=\"https://colab.research.google.com/github/imadelh/NLP-news-classification/blob/master/ULMFIT/01-ulmfit_balanced_dataset.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>" - Train and save ULMFit on full dataset

Notebook contents:
- data preparation
- Fine-tune ULMFit
- Train ULMFit classifier
- Predictions and evaluation
- Exporting the trained model
- Inference on CPU  

Performance of ULMFit on the test dataset `data/dataset_inference` (see end of `02_ulmfit_all_data.ipynb` for the definition of test dataset).

```
# ULMFit - Performance on test dataset
            precision    recall  f1-score   support
micro avg                           0.73     20086
macro avg       0.66      0.61      0.63     20086
weighted avg    0.72      0.73      0.72     20086

Top 3 accuracy on test dataset:
0.9044
```

Trained model can be downloaded from here:
