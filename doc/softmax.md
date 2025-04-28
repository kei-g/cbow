# Softmax

$\mathrm{softmax}$ 関数は以下のように定義される。
ただし、$`S_x(N)=\displaystyle\sum_{n=1}^N e^{\displaystyle x_n}`$ とする。

```math
\mathrm{softmax}_a(x_1,\cdots,x_N)=\frac{\displaystyle e^{\displaystyle x_a}}{\displaystyle S_x(N)}=e^{\displaystyle x_a}{\Bigl(S_x(N)\Bigr)}^{-1}
```

## Partial differentials

$\mathrm{softmax}_a(x_1,\cdots,x_N)$ を $x_b$ で偏微分する。

$a=b$ の場合

```math
\begin{align*}
\frac{\displaystyle\partial\,\mathrm{softmax}_a(x_1,\cdots,x_N)}{\displaystyle\partial\,x_b} &= \frac{\displaystyle\partial\,\mathrm{softmax}_b(x_1,\cdots,x_N)}{\displaystyle\partial\,x_b} \quad\because a=b \\
&= \frac{\displaystyle\partial e^{\displaystyle  x_b}{\Bigl(S_x(N)\Bigr)}^{-1}}{\displaystyle\partial x_b} \\
&= \frac{d}{d\,x_b}\left\{e^{\displaystyle x_b}{\Bigl(S_x(N)\Bigr)}^{-1}\right\} \\
&= e^{\displaystyle x_b}{\Bigl(S_x(N)\Bigr)}^{-1}-e^{\displaystyle x_b}{\Bigl(S_x(n)\Bigr)}^{-2}e^{\displaystyle x_b} \\
&= \frac{\displaystyle e^{\displaystyle x_b}}{S_x(N)}-{\left(\frac{\displaystyle e^{\displaystyle x_b}}{S_x(N)}\right)}^2 \\
&= \frac{\displaystyle e^{\displaystyle x_b}}{S_x(N)}\left(1-\frac{\displaystyle e^{\displaystyle x_b}}{S_x(N)}\right) \\
&= \mathrm{softmax}_b(x_1,\cdots,x_N)\Bigl(1-\mathrm{softmax}_b(x_1,\cdots,x_N)\Bigr) \\
&= \mathrm{softmax}_b(x_1,\cdots,x_N)\Bigl(1-\mathrm{softmax}_a(x_1,\cdots,x_N)\Bigr) \quad\because a=b \\
\end{align*}
```

$a\neq b$ の場合

```math
\begin{align*}
\frac{\displaystyle\partial\,\mathrm{softmax}_a(x_1,\cdots,x_N)}{\displaystyle\partial\,x_b} &= \frac{d}{d\,x_b}\left\{e^{\displaystyle x_a}{\Bigl(S_x(n)\Bigr)}^{-1}\right\} \\
&= 0{\Bigl(S_x(N)\Bigr)}^{-1}+e^{\displaystyle x_a}{\Bigl(S_x(N)\Bigr)}^{-2}e^{\displaystyle x_b} \\
&= -\frac{e^{\displaystyle x_a}e^{\displaystyle x_b}}{{\Bigl(S_x(N)\Bigr)}^{-1}} \\
&= -\frac{e^{\displaystyle x_a}}{S_x(N)}\frac{e^{\displaystyle x_b}}{S_x(N)} \\
&= -\mathrm{softmax}_a(x_1,\cdots,x_N)\mathrm{softmax}_b(x_1,\cdots,x_N) \\
&= \mathrm{softmax}_b(x_1,\cdots,x_N)\Bigl(0-\mathrm{softmax}_a(x_1,\cdots,x_N)\Bigr) \\
\end{align*}
```

クロネッカーのデルタ $\delta_{a,b}$ を用いてまとめると、

```math
\frac{\displaystyle\partial\,\mathrm{softmax}_a(x_1,\cdots,x_N)}{\displaystyle\partial\,x_b}=\mathrm{softmax}_b(x_1,\cdots,x_N)\Bigl(\delta_{a,b}-\mathrm{softmax}_a(x_1,\cdots,x_N)\Bigr)
```

```math
\delta_{a,b}=\begin{cases}
1 & if\;\;a=b \\
0 & otherwise \\
\end{cases}
```
