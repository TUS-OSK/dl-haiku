# dl-haiku

理大祭制作
俳句の生成をやることになった

## Requirements

- [Pipenv](https://pipenv-ja.readthedocs.io/ja/translate-ja/install.html#installing-pipenv)
- [pyenv](https://github.com/pyenv/pyenv) or [Python 3.7](https://www.python.org/downloads/)

## Setup

```sh=
git clone git@github.com:TUS-OSK/dl-haiku.git
cd dl-haiku
pipenv sync --dev
```

make dataset

```sh=
mkdir datasets
python make_simplified_datasets.py -o datasets/train.csv
python make_simplified_datasets.py -n 100 -o datasets/val.csv
```

## Run

```sh=
pipenv run start
```

## Development

### 開発時

以下のコマンドで開発用の Setup を行います
`pipenv install --dev`

### lint

以下のコマンドで，構文エラーのチェックを行います
`pipenv run lint`

### format

以下のコマンドで，Python ファイルのフォーマットを行います
`pipenv run format`

## 自己紹介

- ilim です
- charon です
- Mashu(fymeu) です。18 才です
- muhiyuyuyu ですこんにちは
