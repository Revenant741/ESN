
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    vim \
    language-pack-ja \
    git \
    curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/ \
 && mkdir /root/work

WORKDIR /root

# 文字コードを日本語とutf-8に設定
RUN echo 'export LANG=ja_JP.UTF-8' >> ~/.bashrc \
 && echo 'export LANGUAGE="ja_JP:ja"' >> ~/.bashrc

# vimの環境構築
RUN mkdir -p ~/.vim/pack/git-plugins/start \
 && git clone --depth 1 \
        https://github.com/dense-analysis/ale.git ~/.vim/pack/git-plugins/start/ale \
 && mkdir -p ~/.vim/autoload ~/.vim/bundle \
 && curl -LSso ~/.vim/autoload/pathogen.vim https://tpo.pe/pathogen.vim \
 && cd ~/.vim/bundle \
 && git clone https://github.com/dense-analysis/ale.git \
 && mkdir -p ~/.latte488/ \
 && cd ~/.latte488/ \
 && git clone https://github.com/latte488/configs.git \
 && cp ~/.latte488/configs/.vimrc ~/

# python
RUN pip install flake8 \
 && pip install tensorflow \
 && pip install matplotlib

WORKDIR /root/work

CMD ["/bin/bash", "--login"]

