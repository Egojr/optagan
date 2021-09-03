# OptAGAN

This repository contains the code of the paper: [OptAGAN: Entropy-based finetuning on text VAE-GAN](https://arxiv.org/abs/2109.00239)

OptAGAN stands for Opt(imus) A(ugmented) GAN. It provides a tool for both unconditional and conditional text generation, using the Optimus VAE model, along with a GAN structure and entropy regularized RL finetuning.

### Installation
To reproduce the results presented in the paper, first clone the repository in your environment and install the requirements. 
The file of the requirements was generated through pipreqs. Our enviroment in Google Colab only required the boto3, sacremoses, pytorch-transformers, tensorboardX and transformers==2.6.0 packages.

`git clone https://github.com/Egojr/optagan.git`

`cd optagan/`

`pip install -r requirements.txt`

### How to use

Download a pretrained model from the [Optimus repository.](https://github.com/ChunyuanLI/Optimus)
Then finetune the model on the dataset of choice:

`cd optagan/`

    python run_lm_vae_training.py \
        --output_dir=path/to/finetuned/model/directory \
        --dataset EMNLP \
        --encoder_model_type=bert \
        --encoder_model_name_or_path=bert-base-cased \
        --decoder_model_type=gpt2 \
        --decoder_model_name_or_path=gpt2 \
        --beta 0 \
        --ratio_zero 0.5 \
        --ratio_increase 0.25 \
        --do_train \
        --fb_mode 1 \
        --dim_target_kl 0.5\
        --train_data_file=path/to/train/data/file.txt \
        --eval_data_file=path/to/validation/data/file.txt \
        --num_train_epochs 1.0 \
        --save_steps 10000 \
        --logging_steps 1000 \
        --overwrite_output_dir \
        --per_gpu_train_batch_size=5 \
        --block_size 100 \
        --length_weighted_loss \
        --use_pretrained_model \
        --checkpoint_dir=path/to/pretrained/model/step508523 \
        --latent_size 768 \
        --gloabl_step_eval 508523


Then run the code to train the GAN and finetune the decoder as:

    python optagan.py \
        --dataset EMNLP \
        --checkpoint_dir=path/to/finetuned/model \
        --output_dir=path/to/GAN/directory \
        --encoder_model_type=bert \
        --encoder_model_name_or_path=bert-base-cased \
        --decoder_model_type=gpt2 \
        --decoder_model_name_or_path=gpt2 \
        --train_data_file=path/to/train/data/file.txt \
        --valid_data_file=path/to/validation/data/file.txt \
        --per_gpu_train_batch_size 256 \
        --block_size 100 \
        --max_seq_length 50 \
        --gloabl_step_eval 10000 \
        --latent_size 768 \
        --block_dim 100 \
        --n_layers 10 \
        --interval 50 \
        --epochs 50 \
        --finetune_decoder True \
        --lr_rl 1e-6 \
        --epochs_rl 1000 \
        --batch_size_rl 32
    
    
### Generating sentences

The code to generate sentences and optionally save them to a file is:

    python wgan_test.py \
        --checkpoint_dir=path/to/finetuned/model \
        --output_dir=path/to/output/directory \
        --generator_dir=path/to/GAN/model \
        --block_size 100 \
        --max_seq_length 60 \
        --gloabl_step_eval 10000 \
        --latent_size 768 \
        --block_dim 100 \
        --new_sent 10000 \
        --n_layers 10 \
        --top_p 0.9 \
        --output_name=results \
        --save True
    
### Evaluation

Evaluation of the results can be performed by using the two simple and easy to use scripts: `get_fid.py` and `get_bleu.py`. To calculate the FID score, however, the GLOVE embeddings and the InferSent model are required, and they can be downloaded from [InferSent repository.](https://github.com/facebookresearch/InferSent)


### References

We adapt most of our code from [Optimus repository](https://github.com/ChunyuanLI/Optimus) and take valuable insights and the value head idea from [trl repository.](https://github.com/lvwerra/trl/)
