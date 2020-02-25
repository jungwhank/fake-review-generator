# fake-review-generator

## How to use

### 0. Requirements

We gonna use :hugs: [Hugging Face's Transformers](https://github.com/huggingface/transformers) for text generation.  
Also, you first need to install one of, or both, TensorFlow 2.0 and PyTorch. Please refer to [TensorFlow installation page](https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available) and/or [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.  
Below command is for **Mac**, so check the link above if you use Windows or other OS.

```
pip install torch torchvision
pip install tensorflow
pip install transformer
```

<br>

### 1. Download Dataset

You can download Yelp Dataset in [this link](https://www.yelp.com/dataset/challenge) after you agree to the Dataset License. We gonna use **business.json** and **review.json** file for text generation. Just unzip dataset file in the same directory with this repo.  
<br>

### 2. Preprocess

The file location be like bleow, if you just unzip dataset file in the repo. If you want to change directory,  change some directory in **preprocess.py**.

```
your current directory + /yelp_dataset/business.json
your current directory + /yelp_dataset/review.json
```

**preprocess.py** file extracts only restaurant reviews and divides reviews into positive / neutral / negative reviews. Postivie reviews are 5, 4 star-ratings and Negative reviews are under 2 star-ratings. After preprocessing, output file will be located in this directory.

```
your current directory + /yelp_dataset/preprocess/
```

<br>

### 3. Fine Tuning GPT-2

I modify some code of Hugging Face's [`run_lm_finetuning.py`](https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py).  You can fine-tuning GPT-2 based on positive reviews like this.  It takes many hours depends on your computer.  
To train faster, change args ```--per_gpu_train_batch_size``` (default is 1).

```
python gpt2_fine_tuning.py \
    --output_dir=positive \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file='./yelp_dataset/preprocess/pos.txt'
```

For negative reviews, like below.

```
python gpt2_fine_tuning.py \
    --output_dir=negative \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file='./yelp_dataset/preprocess/neg.txt'
```

<br>

### 4. Generate Fake Reviews

Now, let's generate fake reviews using our model.  
I modify some code of Hugging Face's [`run_generation.py`](https://github.com/huggingface/transformers/blob/master/examples/run_generation.py).  
You can generate positive reviews like this.  

```
python run_generation.py \
    --model_name_or_path=positive \
    --length=100 \
    --seed=42
```

For negative reviews, like below.

```
python run_generation.py \
    --model_name_or_path=negative \
    --length=100 \
    --seed=42
```

To generate different reviews, change args ``--seed`` and `` --length``.

<br>

### 5. Check the Output

For the input 'Price was' and 'Food was', I got positive fake reviews like this

```
Price was a great value and definitely worth a try. Be careful on the parking but still worth it. Again, I'm in the mood for!

Food was great, and the food we ordered was perfect. I love the live entertainment they have.
```

but for negative fake reviews,

```
Price was a good deal and clean but a little while after finishing the meal we got sick from the food. I've had an awful dining!

Food was good, but the food in the kitchen was pretty bland. The duck dumplings were also not hing special.
```

As I mentioned above, change args ``--seed`` and `` --length`` for different reviews.
<br>

### Reference

[Hugging Face's Transformers](https://github.com/huggingface/transformers)

[yelp/dataset-examples](https://github.com/Yelp/dataset-examples)

