# 恋恋日2022 \& こいしの日2022 \& Koishi Day 2022

第14个恋恋日献礼 东方歌词生成器+高性能平行翻译器\
14目のこいしの日おめでとう～東方歌詞ジェネレーターとパラレルトランスレーター\
Present for Koishi Day 2022 - Paralleled Touhou Lyric Generator and Translator (PaToL-GT)
## 使用方法 \& 使い方 \& How to use?
首先，从Google Drvive下载模型状态 \
まず、グーグルドライブからモデルステートをダウンロードしてください \
First, download the model state from Google Drive 
```sh
$ wget https://drive.google.com/file/d/1BmD4PY3YXeMx1HHUVlV8iU2q1_uv2L9J/view?usp=sharing
```

如果你无法使用Google Drive请在issue中联系我！
 
然后将下载到的pths文件放置在与models.py及Pipeline.ipynb的同一个文件夹中\
そして、ダウンロードしたpthsファイルをmodels.py及びPipeline.ipynbと同じなフォルダに置いてください\
Then, put the pths files into the same folder that contains models.py and Pipeline.ipynb.

接着，下载requirements.txt中的依存库\
そして、requirements.txt中の依存パッケージをダウンロードしてください\
Then, download the dependent packages in requirements.txt
```sh
$ pip install requirements.txt
```

最后，运行Jupyter Notebook中的代码来生成东方歌词和翻译吧吧！\
最後にJupyter Notebookのセル中のプログラムを実行しましょう！歌詞と翻訳結果がどんどん出てくるでしょう～\
Finally, run the Jupyter Notebook and see the generated lyrics and translated results!
## 作品展示
```
作曲：ZUN    作词：GPT2-Touhou
--------------------------------------------------
歌词： 夜空に咲いたローズの花は
翻译： 夜空中绽放的蔷薇之花
--------------------------------------------------
歌词： とても美しくて眩しすぎてしまう
翻译： 太过美丽而耀眼
--------------------------------------------------
歌词： 朝はきっと寂し過ぎて
翻译： 清晨一定太过寂寞
--------------------------------------------------
歌词： 燃えるような太陽 このまま沈んでくれないかなと
翻译： 如熊熊燃烧一般的太阳还能像这样沉下去吗
--------------------------------------------------
歌词： わたしがこの手に抱かれて
翻译： 我被这双手拥抱
--------------------------------------------------
歌词： 孤独な夜を越し無常に心
翻译： 在孤独的夜晚中无常的心
--------------------------------------------------
歌词： 夜空に融け損なった言葉達
翻译： 在夜空中溶化殆尽的话语
--------------------------------------------------
```

## 更多作品
### 灵梦
```
作曲：ZUN    作词：GPT2-Touhou
--------------------------------------------------
歌词： 博麗の巫女よ 強く照らす月に
翻译： 博丽巫女啊照耀着的月光
--------------------------------------------------
歌词： 髪を川に浮かべ光る暁に
翻译： 在川流的光芒四射的黎明
--------------------------------------------------
歌词： 緋色の華を失くさないように
翻译： 为了不失去绯色之花
--------------------------------------------------
歌词： 獣たちの天を撫でて
翻译： 抚摸着野兽们的天空
--------------------------------------------------
歌词： 溢れる雲をかき分け
翻译： 穿过满溢的云朵
--------------------------------------------------
歌词： 遠く或いは浅き岸を抜け
翻译： 越过遥远又浅浅的岸边
--------------------------------------------------
歌词： 幻想郷を貫いて
翻译： 贯穿幻想乡
--------------------------------------------------
```
### 魔理沙
```
作曲：ZUN    作词：GPT2-Touhou
--------------------------------------------------
歌词： 恋色の星空 手繰り寄せては
翻译： 恋色的星空将我牵起
--------------------------------------------------
歌词： だから素敵な夢を追うの
翻译： 所以追寻着美好的梦想
--------------------------------------------------
歌词： 恋色に射した指先 流す涙のようで
翻译： 被恋色照耀的指尖就像流下的泪水
--------------------------------------------------
歌词： 永遠を求めてくから そこで二人繰り返す時
翻译： 在追求永恒之时在那里两人不断重复
--------------------------------------------------
歌词： 世界はまわる 永遠を重ねてく
翻译： 世界在循环着永远重叠着
--------------------------------------------------
```
### 早苗
```
作曲：ZUN    作词：GPT2-Touhou
--------------------------------------------------
歌词： 涼しい雨上がり 虹への想いに気付いた
翻译： 在冰冷的雨后注意到彩虹的思念
--------------------------------------------------
歌词： 愚かな欲望も引き金と成り
翻译： 就连愚蠢的欲望也成了契机
--------------------------------------------------
歌词： 大空 ひとつの雲は どこまでも聳え立つ
翻译： 天空上一片云朵无论在何处都能看见
--------------------------------------------------
歌词： そりゃ真の自分に気付いたよ
翻译： 我注意到了真正的自己
--------------------------------------------------
歌词： 栄えある 鳥類千里を行く
翻译： 在千里之遥之千里
--------------------------------------------------
歌词： 鳥達は 空を飛び自由に飛翔楽園
翻译： 鸟儿们自由飞翔自由飞翔的乐园
--------------------------------------------------
```
### 觉大人
```
作曲：ZUN    作词：GPT2-Touhou
--------------------------------------------------
歌词： 地霊殿に踊る少女は
翻译： 地灵殿中起舞的少女
--------------------------------------------------
歌词： 吸血の凶器にまみれた
翻译： 被吸血的凶器包围
--------------------------------------------------
歌词： 魂を蝕む悲鳴は
翻译： 侵蚀灵魂的悲鸣
--------------------------------------------------
歌词： 奈落の魔獣を仰ぐように
翻译： 就像是仰望奈落魔兽一般
--------------------------------------------------
歌词： 地獄の涅槃 煉獄から手招き
翻译： 从地狱的涅槃来邀请我
--------------------------------------------------
歌词： 思えば刃向かう 燃え盛る賜物よ
翻译： 回想起来就刃刃而熊熊燃烧
--------------------------------------------------
```
### 蕾米莉亚
```
作曲：ZUN    作词：GPT2-Touhou
--------------------------------------------------
歌词： 緋色の主は黒須の姫を 鮮やかに掻きゆし刻んでゆく
翻译： 绯色之主将黑色之姬鲜明地铭刻于心
--------------------------------------------------
歌词： 子供の頃に聞いたあなたの声さえいまは懐かしい
翻译： 小时候所听到的你的声音如今也令人怀念
--------------------------------------------------
歌词： 遠い記憶を追いかけていたから
翻译： 追逐着遥远的记忆
--------------------------------------------------
歌词： あなたは私を探してる 本当は胸を焦がしていた
翻译： 你一直在寻找着我其实是焦灼着我的心
--------------------------------------------------
歌词： 探し続けていたの まぶたの奥の奥の奥
翻译： 一直在寻找着啊在眼眸深处的深处
--------------------------------------------------
```
### 芙兰朵露
```
作曲：ZUN    作词：GPT2-Touhou
--------------------------------------------------
歌词： すべてを壊したら
翻译： 如果将一切破坏的话
--------------------------------------------------
歌词： 舞える星座に恋をこめて
翻译： 将恋恋于绽放的星座
--------------------------------------------------
歌词： 逢いたいんだ 揺れるあいちゃん
翻译： 好想见你摇摆不定的小幸
--------------------------------------------------
歌词： 届いてますか 勇気をください
翻译： 传达到了吗给我勇气
--------------------------------------------------
歌词： 残酷なほど美しい空模様にうとんで
翻译： 在那残酷而美丽的天空中
--------------------------------------------------
歌词： 闇の中では芽生えた恋
翻译： 在黑暗中萌芽的恋爱
--------------------------------------------------
歌词： ピンク色した世界で
翻译： 在红色的世界里
--------------------------------------------------
歌词： 叫ぶ君の声は
翻译： 你呼喊的声音
--------------------------------------------------
```
### 琪露诺
```
作曲：ZUN    作词：GPT2-Touhou
--------------------------------------------------
歌词： 賢い氷の妖精 煌き知る星の
翻译： 聪明的冰之妖精闪闪发光的星星
--------------------------------------------------
歌词： 砕け散れ
翻译： 破碎散落
--------------------------------------------------
歌词： 光の輪くぐり 空に宙還る月
翻译： 穿过光之轮，在天空中空回月
--------------------------------------------------
歌词： 見て見て 星空も花咲かす
翻译： 快看快看星空也会绽放花朵
--------------------------------------------------
歌词： 月明かりの下 パチリ
翻译： 在月光之下拍拍拍
--------------------------------------------------
歌词： 君の姿を 星の海に広がる
翻译： 你的身影在星之海蔓延
--------------------------------------------------
歌词： 煌めく毬色の雫は
翻译： 闪闪发光的花珠
--------------------------------------------------
```
