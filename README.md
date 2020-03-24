# ESN

## 実行するまでの手順

### 環境構築
1. まずクローンします.\
`git clone https://github.com/latte488/ESN.git`
1. クローンしたディレクトリに移動します.\
`cd ESN`
1. 次にdockerのimageを作成します.\
`docker build -t esn:20.03 .`
1. 次にdockerのコンテナを作成します. 
このdockerfileの環境では`/root/projects`ディレクトリが作成されるためそこにマウントします.\
``docker run -v `pwd`:/root/projects -it --name esn esn:20.03``
1. 最後にコンテナの中から抜けます
`exit`

### 実行例(reservoir_test.py)
1. まず`esn`コンテナを起動させます.\
`docker start esn`
1. 次に`esn`コンテナに入ります.\
`docker attach esn`
1. 次に`projects/`に移動します\
`cd projects`
1. 最後にpythonを実行します \
`python reservoir_test.py`

### 注意点
* コンテナに入るときにrootで入るためコンテナ内でファイルを作成すると戻った時に権限がrootになってしまいます.
  * 解決策は外でコーディングして実行時のみコンテナで行うことです.
  * gitの設定も毎回する必要があるから基本作業はコンテナ外がベストなのかもしれません.
