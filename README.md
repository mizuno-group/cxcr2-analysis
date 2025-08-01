# 論文を表すプロジェクト名
論文公開用のtemplate repository  
git cloneしたら再現できることを目指す.  

# ！最初にやること！
- READMEの変更  
    - repository nameの変更  
    - Authorsの追記  
    - Contactの追記  
- notebooks内の変更  
    - 論文データの再現用にsample.ipynbを置く  
    - 適宜分ける, figureごとに分けてもいい  
    - 必要に応じてsrcにソースコード, dataに使用データを置く  
    - モデル開発の場合には, template_pkgを参照してsetup.pyを置いてパッケージ化する  

# Note
This repository is under construction and will be officially released by [Mizuno group](https://github.com/mizuno-group).  
Please contact tadahaya[at]gmail.com before publishing your paper using the contents of this repository.  

## Publication
- [peer-reviewed publication](https://academic.oup.com/bib/article/25/4/bbae315/7709575)  
- [preprint](https://www.biorxiv.org/content/10.1101/2024.01.08.574749v2)  

## Organization
------------  

    ├── LICENSE  
    │
    ├── README.md            <- The top-level README for developers using this project.  
    │
    └── notebooks            <- Notebooks for reproducibility.
        │
        ├── data             <- Data employed in this study.
        │
        ├── src              <- Source code employed in this study.
        │  
        └── sample.ipynb     <- ipynb employed in this study.

------------

## Authors
- [自身の名前](自身のgithub repository)  
    - main contributor  
- [Tadahaya Mizuno](https://github.com/tadahayamiz)  
    - correspondence  

## Contact
If you have any questions or comments, please feel free to create an issue on github here, or email us:  
- [自身のアドレス]  
- tadahaya[at]gmail.com  
    - lead contact  