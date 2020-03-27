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
docker start esn
docker attach esn
cd projects
python reservoir_test.py
```

## このdockerの環境のアンインストール
```bash
python3 clean.py
```
