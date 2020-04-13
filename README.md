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
(base imageは削除されません)
```bash
python3 clean.py
```

Leakyの適正値

１：通常ESNと同じ状態　精度：50％

0.9：ほぼESNと同じ，少しだけ今までの関数を考慮する　精度：9％　何故かは不明

0.8：    精度：55％

0.7：主に直前の入力を考えるが，今までの関数も考慮する　精度：60％

0.6：

0.5：直前の入力と今までの関数を半分づつ考慮する　精度：９％　なぜかは不明

0.02：直前の入力はほぼ考慮しない　精度：40％
