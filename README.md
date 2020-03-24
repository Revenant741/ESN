# ESN

## 実行するまでの手順

### 環境構築
```bash
git clone https://github.com/latte488/ESN.git
cd ESN
docker build -t esn:20.03 .
docker run -v `pwd`:/root/projects -it --name esn esn:20.03
exit
```
### 実行例(reservoir_test.py)
```bash
docker start esn
docker attach esn
cd projects
python reservoir_test.py
```

### 注意点
* コンテナに入るときにrootで入るためコンテナ内でファイルを作成すると戻った時に権限がrootになってしまいます.
  * 解決策は外でコーディングして実行時のみコンテナで行うことです.
  * gitの設定も毎回する必要があるから基本作業はコンテナ外がベストなのかもしれません.
