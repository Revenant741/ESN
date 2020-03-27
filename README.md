# ESN

## 環境構築

`toml`ファイルをpythonで扱うためモジュールがない場合はインストールします.
```bash
pip3 install toml
```

`config.toml`に基づいてdockerのimageとcontainerを作成します.
```bash
git clone https://github.com/latte488/ESN.git
cd ESN
python3 setup.py
```
## 実行例(reservoir_test.py)
```bash
docker start -i esn
python src/reservoir_test.py
```

## このdockerの環境のアンインストール

dockerのimageとcontainerの削除と`docker image prune`を行います.
```bash
python3 clean.py
```
