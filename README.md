# Continuous Bag-of-Words

## Concept

語彙数を $V$ とする。

$x\in[1,V]$ 番目の単語の one hot vector $\overrightarrow{w_x}$ は $x$ 番目の要素だけが $1$ の縦ベクトルで、次のように表せる。

```math
\overrightarrow{w_x}=\begin{pmatrix}
0 \\
\vdots \\
1 \\
\vdots \\
0 \\
\end{pmatrix}
```

### Input layer

$t\in(c,V-c)$ 番目の単語の前後 $c$ 個の単語の one hot vector について、
それぞれに $h\times V$ の行列 $I$ を掛けて $2c$ 個の $h$ 次元ベクトルを得る。

行列 $I$ と $\overrightarrow{w_x}$ との積 $u_{1,x}$ は、

```math
u_{1,x}
=
\begin{pmatrix}
I_{1,1} & I_{1,2} & \cdots & I_{1,V} \\
\vdots \\
I_{h,1} & \cdots & & I_{h,V} \\
\end{pmatrix}
\overrightarrow{w_x}
=
\begin{pmatrix}
I_{1,x} \\
\vdots \\
I_{h,x} \\
\end{pmatrix}
```

$2c$ 個の $h$ 次元ベクトルの平均値をとり、入力層の出力ベクトル $\overrightarrow{z_1}$ を得る。

```math
\overrightarrow{z_1}
=
\frac{1}{2c}\overrightarrow{u_1}
=
\frac{1}{2c}
\begin{pmatrix}
\displaystyle\sum_{k=t-c}^{t-1}I_{1,k}+\sum_{k=t+1}^{t+c}I_{1,k} \\
\vdots \\
\displaystyle\sum_{k=t-c}^{t-1}I_{h,k}+\sum_{k=t+1}^{t+c}I_{h,k} \\
\end{pmatrix}
=
\begin{pmatrix}
z_{1,1} \\
z_{1,2} \\
z_{1,3} \\
z_{1,4} \\
\vdots \\
z_{1,h} \\
\end{pmatrix}
```

### Output layer

このベクトル $\overrightarrow{z_1}$ に $V\times h$ の行列 $O$ を掛けると、次のベクトル $\overrightarrow{u_2}$ が得られる。

```math
\overrightarrow{u_2}
=
\begin{pmatrix}
O_{1,1} & \cdots & O_{1,h} \\
O_{2,1} & \cdots & O_{2,h} \\
\vdots & & \\
O_{V,1} & \cdots & O_{V,h} \\
\end{pmatrix}
\overrightarrow{z_1}
=
\begin{pmatrix}
\displaystyle\sum_{k=1}^h{O_{1,k}z_{1,k}} \\
\vdots \\
\displaystyle\sum_{k=1}^h{O_{V,k}z_{1,k}} \\
\end{pmatrix}
=
\begin{pmatrix}
u_{2,1} \\
u_{2,2} \\
u_{2,3} \\
u_{2,4} \\
\vdots \\
u_{2,V} \\
\end{pmatrix}
```

$\overrightarrow{u_2}$ を $\mathrm{softmax}$ で活性化して出力ベクトル $\overrightarrow{z_2}$ を得る。

```math
\overrightarrow{z_2}
=
\begin{pmatrix}
\mathrm{softmax}_1(u_{2,1},\cdots,u_{2,V}) \\
\vdots \\
\mathrm{softmax}_V(u_{2,1},\cdots,u_{2,V}) \\
\end{pmatrix}
=
\begin{pmatrix}
z_{2,1} \\
\vdots \\
z_{2,V} \\
\end{pmatrix}
```

## Learning

$\overrightarrow{z_2}$ と $\overrightarrow{w_t}$ の誤差（損失）を減らすよう学習を進める。

損失関数を $E_t(\overrightarrow{z_2})$ として、 $`f(x)=\displaystyle\frac{\displaystyle\partial\,E_t(\overrightarrow{z_2})}{\displaystyle\partial\,u_{2,x}}`$ とおくと、

```math
\begin{align*}
f(x) &= \sum_{n=1}^V\frac{\displaystyle\partial\,E_t(\overrightarrow{z_2})}{\displaystyle\partial\,z_{2,n}}\frac{\displaystyle\partial\,z_{2,n}}{\displaystyle\partial\,u_{2,x}} \\
&= \sum_{n=1}^V\frac{\displaystyle\partial\,E_t(\overrightarrow{z_2})}{\displaystyle\partial\,z_{2,n}}\frac{\displaystyle\partial\mathrm{softmax}_n(u_{2,1},\cdots,u_{2,V})}{\displaystyle\partial\,u_{2,x}} \\
&= \sum_{n=1}^V\left(\frac{d}{d\,z_{2,n}}E_t(z_{2,n})\right)z_{2,x}\bigl(\delta_{n,x}-z_{2,n}\bigr) \\
&= z_{2,x}\left(\sum_{n=1}^V\left(\frac{d}{d\,z_{2,n}}E_t(z_{2,n})\right)\delta_{n,x}-\sum_{n=1}^V\left(\frac{d}{d\,z_{2,n}}E_t(z_{2,n})\right)z_{2,n}\right) \\
&= z_{2,x}\left(\left(\frac{d}{d\,z_{2,x}}E_t(z_{2,x})\right)-\sum_{n=1}^V\left(\frac{d}{d\,z_{2,n}}E_t(z_{2,n})\right)z_{2,n}\right) \\
\end{align*}
```

$`E^{\prime}_t(x)=\displaystyle\frac{d}{d\,z_{2,x}}E_t(z_{2,x})`$ とおくと、

```math
f(x)=z_{2,x}\left(E^{\prime}_t(x)-\sum_{n=1}^V E^{\prime}_t(n)z_{2,n}\right)
```

### for the matrix $O$

損失関数 $E_t(\overrightarrow{z_2})$ を行列 $O$ の各要素で偏微分したものに学習係数 $\eta$ を掛けた数値でその要素を更新することで、損失は小さくなる。

行列 $O$ の $i$ 行 $j$ 列の更新量を $\Delta\,O_{i,j}$ とすると、

```math
\begin{align*}
\Delta\,O_{i,j} &= -\eta\frac{\displaystyle\partial\,E_t(\overrightarrow{z_2})}{\displaystyle\partial\,O_{i,j}} \\
&= -\eta\sum_{n=1}^V\frac{\displaystyle\partial\,E_t(\overrightarrow{z_2})}{\displaystyle\partial\,u_{2,n}}\frac{\displaystyle\partial\,u_{2,n}}{\displaystyle\partial\,O_{i,j}} \\
&= -\eta\sum_{n=1}^V f(n)\frac{\displaystyle\partial\sum_{k=1}^h O_{n,k}z_{1,k}}{\displaystyle\partial\,O_{i,j}} \\
&= -\eta\,f(i)\,z_{1,j} \\
\end{align*}
```

### for the matrix $I$

損失関数 $E_t(\overrightarrow{z_2})$ を行列 $I$ の各要素で偏微分したものに学習係数 $\eta$ を掛けた数値でその要素を更新することで、損失は小さくなる。

行列 $I$ の $i$ 行 $j$ 列の更新量を $\Delta\,I_{i,j}$ とすると、

```math
\begin{align*}
\Delta\,I_{i,j} &= -\eta\frac{\displaystyle\partial\,E_t(\overrightarrow{z_2})}{\displaystyle\partial\,I_{i,j}} \\
&= -\eta\sum_{n=1}^V\frac{\displaystyle\partial\,E_t(\overrightarrow{z_2})}{\displaystyle\partial\,u_{2,n}}\frac{\displaystyle\partial\,u_{2,n}}{\displaystyle\partial\,I_{i,j}} \\
&= -\eta\sum_{n=1}^V f(n)\sum_{k=1}^h\frac{\displaystyle\partial\,u_{2,n}}{\displaystyle\partial\,z_{1,k}}\frac{\displaystyle\partial\,z_{1,k}}{\displaystyle\partial\,I_{i,j}} \\
&= -\eta\sum_{n=1}^V f(n)\sum_{k=1}^h\frac{\displaystyle\partial\sum_{s=1}^h O_{n,s}z_{1,s}}{\displaystyle\partial\,z_{1,k}}\frac{\displaystyle\partial\frac{1}{2c}\left(\sum_{s=t-c}^{t-1}I_{k,s}+\sum_{s=t+1}^{t+c}I_{k,s}\right)}{\displaystyle\partial\,I_{i,j}} \\
\end{align*}
```

指示関数 $\mathbb{I}_t(x)$ を次のように定義すると、

```math
\mathbb{I}_t(x)=\begin{cases}
1 & if\;x\in[t-c,t-1]\cup[t+1,t+c] \\
0 & otherwise \\
\end{cases}
```

```math
\Delta\,I_{i,j}=-\eta\frac{\mathbb{I}_t(j)}{2c}\sum_{n=1}^V f(n)O_{n,i}
```

### Cross Entropy Loss

$E_t(\overrightarrow{z_2})$ を交差エントロピー損失関数とすると、

```math
\begin{align*}
E_t(\overrightarrow{z_2}) &= -\sum_{n=1}^V w_{t,n}\log(z_{2,n}) \\
E^{\prime}_t(x) &= -\frac{w_{t,x}}{z_{2,x}} \\
f(x) &= z_{2,x}\left(-\frac{w_{t,x}}{z_{2,x}}-\sum_{n=1}^V\left(-\frac{w_{t,n}}{z_{2,n}}\right)z_{2,n}\right) \\
&= -w_{t,x}+z_{2,x}\sum_{n=1}^V w_{t,n} \\
&= -w_{t,x}+z_{2,x}\cdot 1 \\
&= z_{2,x}-w_{t,x} \\
\Delta\,O_{i,j} &= -\eta\bigl(z_{2,i}-w_{t,i}\bigr)z_{1,j} \\
\Delta\,I_{i,j} &= -\eta\frac{\mathbb{I}_t(j)}{2c}\sum_{n=1}^V\bigl(z_{2,n}-w_{t,n}\bigr)O_{n,i} \\
\end{align*}
```

### Word vector

行列 $I$ の各列が、対応する各単語の単語ベクトルとなる。

## Usage

```text
cbow [<オプション>]... [<ファイル>]

空白・ハイフン・改行で区切られた英文を学習して、単語ベクトルを JSON 形式で出力します。

オプション:
  -c, --case-sensitive  大文字小文字を区別する
                        (デフォルト: 区別しない)
  -d, --dimension=uint  単語ベクトルの次元
                        (デフォルト: 128)
  -e, --eta=float       学習係数
                        (デフォルト: 0.001953125)
  -m, --min-count=uint  単語の最小出現頻度
                        (デフォルト: 5)
  -v, --verbosity=int   表示内容の組み合わせ
                          0: 進捗と単語ベクトル
                          1: 単語数と語彙数
                          2: 単語の出現頻度
                          4: 新出語彙
                          8: 既出語彙
                          16: 訓練中の単語推定情報
                          32: 損失のヒストグラム
                        (デフォルト: 0)
  -w, --width=uint      周辺単語数
                        (デフォルト: 4)

<ファイル>が指定されていない場合は標準入力から読み込みます。

単語ベクトルは標準出力へ、それ以外は標準エラー出力へ出力されます。

損失のヒストグラムを出力するには sixel に対応している terminal が必要です。
```
