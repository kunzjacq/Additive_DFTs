# Additive DFT algorithms: Cantor, Mateer-Gao

Note: this document is best viewed oustide of GitHub which has several LaTeX rendering bugs.

## References

- [1] Todd Mateer's master thesis, "Fast Fourier Transform Algorithms with Applications", 2008.

## Von Zur Gathen - Gerhard, Cantor - Wang Zhu

### Von Zur Gathen - Gerhard

- $\mathbb{F}$ is the finite field of size $2^n$,
- $\{\beta_1, \ldots, \beta_n\}$ is a basis of $\mathbb{F}$ on $\mathbb{F}_2$,
- $W_0=\{0\}$, $W_i=<\beta_1, \ldots, \beta_i>$, $C_i=<\beta_{i+1}, \ldots, \beta_n>$, $i=1\ldots n$,
- $\sigma_i$ is the polynomial that cancels on $W_i$:
$$\sigma_i = \prod_{a\in W_i} (X-a)$$
then $\deg \sigma_i = 2^i$ and

$$\sigma_0(X) = X,$$

$$\sigma_{i+1}(X) = \sigma_i(X-\beta_{i+1}) \: \sigma_i(X). \quad (R_0)$$

The latter results from $W_{i+1}\setminus W_i = W_i + \beta_{i+1}$.

It is shown recursively that the functions associated to $\sigma_{i}$ are linear ($\sigma_i$ only has monomials whose degree is a power of 2) and that 

$$\sigma_{i+1}(X) = \sigma_i^2(X) - \sigma_i(X)\beta_{i+1}. \quad (R)$$

(apply $\sigma_i$ linearity to $(R_0)$).

By the linearity of $\sigma_i$, for $\varepsilon \in C_{i+1}$, $\sigma_i(x - \beta_{i+1} - \varepsilon) = \sigma_i(x) - \sigma_i(\beta_{i+1} + \varepsilon)$. $\sigma_i$ has at most $i$ non-zero terms (corresponding to the exponents which are powers of 2, excluding the constant term).

This suffices to deduce the Von Zur Gathen - Gerhard DFT algorithm in $\mathbb{F}$:

Given a polynomial $f$ on $\mathbb{F}$, $\varepsilon \in C_{i+1}$, and $g = f \mod \sigma_{i+1}(x-\varepsilon)$, if $g_0 = g \mod \sigma_i(x-\varepsilon)$ and $g_1 = g \mod \sigma_i(x-\varepsilon - \beta_{i+1})$, then with $g = g_0 + \sigma_i(x-\varepsilon) q$, $\deg(h_0) < 2^i$,
$$g_1 = g_0 + \sigma_i(\beta_{i+1}) q$$
Indeed, $g = g_0 + \sigma_{i+1}(x-\varepsilon) q = [g_0 + \sigma_i(\beta_{i+1}) q] + \sigma_i(x-\varepsilon - \beta_{i+1}) q$.

Starting from $g$, we can thus compute $g_0$ and $g_1$ for the price of a **single Euclidean division** followed by the addition of $q \times \text{constant}$ to $g_0$. This is the core algorithmic trick that yields the alogrithmic complexity of Von Zur Gathen - Gerhard.

The Von Zur Gathen - Gerhard algorithm consists in performing this computation starting from a polynomial of degree $<2^n$ on $\mathbb{F}$, for $i=n-1,\ldots,0$, with $2^{n-1-i}$ Euclidean divisions at each step, corresponding to all possible offsets $\varepsilon$. The remainders of the Euclidean divisions at the last step are nothing else than the values of $f$ at the corresponding $\varepsilon$ points. The algorithm performs $O(k \log^2(k))$ additions or multiplications, $k=2^n$. See [1] p.99 for a precise analysis of the number of additions and multiplications performed in $\mathbb{F}$.

### Improvement: Cantor - Wang Zhu, Cantor bases

The Von Zur Gathen - Gerhard algorithm does not make any assumption about the basis $\{\beta_1,\ldots,\beta_n\}$ used. Choosing the right basis allows to reduce the number of multiplications to be performed by making sure that the $\sigma_i$ have only coefficients equal to 1, and that for any $i$, $\sigma_i(\beta_{i+1}) = 1$. For this, we choose the $\beta_i$ s.t.
$$\forall i < n, \beta_i = \beta_{i+1}^2 + \beta_{i+1} \quad (E)$$

The  basis obtained is a *Cantor basis*. Such a basis exists only if $n$ is a power of two; let's write $n=2^s$. See below for a justification of this and the construction of the basis.

With a Cantor basis, it is shown by recurrence from $(R)$ that

$$\forall i,\ \sigma_i(\beta_{i+1}) = 1 \text{ and } \sigma_{i+1} = \sigma_i^2 + \sigma_i = \varphi(\sigma_i) \text{ with }\varphi(x) = x^2 + x.$$
Then $\forall i,\ \sigma_i = \varphi^{\circ i}(x)$ and all the non-zero coefficients of the  $\sigma_i$ are equal to 1. This decreases the number of multiplications to be performed during the Euclidean divisions in the Von Zur Gathen - Gerhard algorithm. The number of multiplications to evaluate a polynomial of degree $k$ is in $O(k \log(k))$ and the number of additions becomes $O(k \log^{1.585}(k))$. See [1] p.106 for the calculation of these bounds.

The improved algorithm is called **Cantor** or **Wang-Zhu** algorithm.

Additional properties of Cantor bases are useful for the Mateer-Gao algorithm presented in the next paragraph.

- for $t=2^j \leq n$, $\sigma_{t}=x^{2^t} + x$. This is shown by recurrence:
  - $\sigma_1=x^2+x$;
  - if $\sigma_t = \varphi^{\circ t}(x) = x^{2^t} + x$,
$\sigma_{2t} = \varphi^{\circ 2t} = \varphi^{\circ t}(x^{2^t}+x)= (x^{2^t} + x)^{2^t} + x^{2^t} + x = x^{2^{2t}} + x^{2^t} + x^{2^t} + x = x^{2^{2t}}+x$.
- If $$\omega_j = \sum_{i = 0}^{n-1} \varepsilon_i \beta_{i+1}, \quad \varepsilon_i \in{0,1}$$ with $j$ the integer $\sum_{i = 0}^{n-1} \varepsilon_i \ 2^i,$ then
  $$\sigma_i(\omega_j) = \omega_{j\gg i}$$
  
  where $\gg$ represents the left shift (i.e. $j \gg i$ = $j/2^i$, truncated to the lower integer). This follows from
  
  - the linearity of $\sigma_i = \varphi^{\circ i}$
  - the fact that $\forall i>1, \varphi(\beta_i) = \beta_{i-1}$, and $\varphi(\beta_1) = \varphi(1) = 0$.

As a particular case, $\sigma_i(\beta_{i+1})=\sigma_i(\omega_{2^{i}})=\omega_{1} = 1$.

Let us also mention the following formula (not used in the following):
$$\sigma_i (x) = \sum_{d=0}^{i} \left(\begin{array}{c}i\\ d\end{array}\right) \ x^{2^d}$$
The $\left(\begin{array}{c}i\\ d\end{array}\right)$ are the binomial coefficients and are understood modulo 2. The formula is shown by recurrence, like the one for the development of $(1+x)^i$.

#### Construction of a Cantor basis

To build such a basis, we can proceed in two ways:

#### "Inner" construction

Starting from a given field $\mathbb{F}$ of cardinal $2^{2^s}$, $n=2^s$, let us notice that:

- $\sigma_n(x) = x^{2^n} + x$ has exactly the $2^n$ elements of $\mathbb{F}$ as roots;
- $\sigma_n = \sigma_{n-i} \circ \sigma_i$; since $\sigma_{n-i}(0)=0$, the roots of $\sigma_{n-i}$ are included in those of $\sigma_n$, so they are all in $\mathbb{F}$; therefore the $2^i$ roots of $\sigma_i$ are in $\mathbb{F}$.

We choose $\beta_n$ a root of $\sigma_n$ but not of $\sigma_{n-1}$, and for all $i < n$, $\beta_i = \varphi(\beta_{i+1})$. Then $\varphi(\beta_1) = \sigma_n(\beta_n) = 0$ but $\beta_1 = \sigma_{n-1}(\beta_n) \neq 0$: as $\varphi(x)=0$ has roots 0 and 1, $\beta_1=1$.

As $\sigma_i$ has $2^i$ roots in $\mathbb{F}$, there are $2^{n-1}$ valid choices for $\beta_n$, that is to say, half of the values of $\mathbb{F}$; it is thus not difficult to find a valid value for $\beta_n$.

Starting from a constructed basis $\{\beta_i\}$, and with $\beta$ a linear combination of $\beta_1,\ldots,\beta_{n-1}$, let $\gamma_n = \beta_n + \beta$ and for all $i < n$, $\gamma_i = \varphi(\gamma_{i+1})$. Then by linearity of $\sigma_n$ and $\sigma_{n-1}$, $\sigma_n(\gamma_n)=0$ and $\sigma_{n-1}(\beta_n) = \sigma_{n-1}(\beta_n) \neq 0$: $\{\gamma_i\}$ is another Cantor basis. All Cantor bases are obtained in this way because there are $2^{n-1}$ possible values for $\beta$.

This construction shows that whenever $n=2^s$, a Cantor basis exists.

#### "Outer" construction: joint construction of the field and the basis

We can also construct $\mathbb{F}$ jointly with $\beta_i$, by successive extensions of $\mathbb{F}_{2}$: if $\beta_1, \beta_2, \ldots,\beta_i$ are already constructed with $\beta_1=1$, $\beta_{i+1}$ is constructed as a solution of $(E)$. If a field of dimension $n={2^s}$ over $\mathbb{F}_2$ has been produced thus far, this construction works for $\beta_1, \ldots, \beta_n$ (see discussion in for the "inner" construction). When there is no solution, the field $<\beta_1,\ldots,\beta_i>$ is extended by an element *defined* by $(E)$, which has the effect of doubling the dimension of $\mathbb{F}$ over $\mathbb{F}_2$.

A related argument shows why, when $\mathbb{F}$ has a Cantor basis, one must have $n=2^s$: if $\beta_1, \ldots, \beta_n$ is a Cantor basis, either $\beta_{i+1}$ belongs to $k=\mathbb{F}_2[\beta_1,\ldots,\beta_{i-1}]$, or it is an extension of degree 2 of $k$. In the end, $\mathbb{F}$ is a tower of extensions of degree 2 of $\mathbb{F}_2$, therefore has degree $2^s$ over it for some $s$.

## Mateer-Gao

### Idea

A Cantor basis is used, we use the same notations as above.

Input: $s \geq 0$, $t = 2^{s-1}$, $\eta = 2^{2t}$, $f$ polynomial of degree $< \eta$, an *offset* $j < 2^{n-2t}$.

Output: in an array of size $\eta$: $\{f(\omega_{\eta j}), f(\omega_{\eta j + 1}), \ldots, f(\omega_{(\eta + 1) j - 1})\}$ = evaluation of $f$ over $W_{2t} + \omega_{\eta j}$.

Let $\tau = 2^t$, $\eta = 2^{2t}$, $2t = 2^s$. Then 

$$\eta = \tau^2,$$
$$\sigma_t(X) = X^{2^t} - X = X^{\tau} - X,$$
$$\sigma_{2t}(X) = X^{2^{2t}} - X = X^{\eta} - X.$$

If $f$ is processed as part of a higher level recursive call, $f$ represents the initial polynomial $f_0$ modulo $\sigma_{2t}(x- \omega_{\eta j})$. $f$ and $f_0$ coincide on $W_{2t} + \omega_{\eta j}$.

The $\sigma_{k}$ are linear and as shown previously,

$$\sigma_{k}(\omega_{j}) = \omega_{j\gg k}.$$

$\sigma_{2t}$ cancels on $W_{2t}$ and $\sigma_{t}$ on $W_t$; the $W_t+\omega_{\tau k}$ partition $W_{2t}$ for $k\leq \tau$, so
$$\sigma_{2t}(X) = \prod_{k < \tau} \sigma_{t}(X - \omega_{\tau k}) = \prod_{k < \tau} \left [\sigma_{t}(X) - \omega_{k} \right ].$$

This relation can be translated by $\omega_{\eta j}$ to give
$$\sigma_{2t}(X - \omega_{\eta j}) = \sigma_{2t}(X) - \omega_j = \prod_{k < \tau} \left[\sigma_{t}(X - \omega_{\eta j}) - \omega_{k} \right] = \prod_{k < \tau} \left[\sigma_{t}(X) - \omega_{\tau j + k} \right].$$

If we deduce from $f = f_0 \mod \sigma_{2t}(x - \omega_{\tau j})$ the set of $f_0$ modulo the $R_k = \sigma_{t}(x) - \omega_{\tau j + k}$, $k < \tau$, it is then possible to finish the evaluation of $f_0$ on  $W_{2t} + \omega_{\eta j}$ by performing a DFT on each of the $R_k$, which will give the values of $f_0$ on the spaces $W_t + \omega_{\eta j + \tau k}$, $k< \tau$, which partition $W_{2t} + \omega_{\eta j}$.

**The efficiency of the algorithm comes from not doing the computations of $f_0$ mod $R_k = \sigma_{t}(X) - \omega_{\tau j + k}$ for $k < \tau$ independently**. Instead, we proceed as follows:

First, $f$ is decomposed as

$$f(X) = \sum_{\ell<\tau} f_{\ell}(X) \: \sigma_{t}^{\ell}$$
with $\forall \ell < \tau, \deg f_{\ell} < \tau$.

(This is a "generalized Taylor decomposition"; it turns out that it can be computed linearly in $\eta$.)

Then, as $R_k = \sigma_{t} - \omega_{\tau j + k}$, $\sigma_t^\ell = \omega_{\tau j + k}^{\ell} \mod R_k$ and

$$f \mod R_k = \sum_{\ell<\tau} f_{\ell} \: \omega_{\tau j + k}^{\ell}.$$

In other words, the $\{f \mod R_k, k < \tau\}$ are evaluations of the polynomial $P(Y)= \sum_{i<\tau} f_i(X) Y^i$ on the points of the space $W_t + \omega_{\tau j}$. These evaluations can therefore be performed by an DFT. The fact that the coefficients of $P$  and the values to compute are themselves polynomials is not a problem: $\tau$ DFT are performed according to the degrees in $X$ and the final values are obtained by reassembling the DFT results.

The recursion step ends with an DFT of each of the $R_k$, which produces the desired values of $f$ (or $f_0$).

This gives in practice the following algorithm:

### Pseudo-algorithm

If $\eta = 2$ (i.e. $t = 1$, $s = 0$), a direct computation is performed ($u = f_0+\omega_{2j}$, $f_0 + \omega_{2j+1} = f_0 +\omega_{2j} + 1 = u + 1$)

Otherwise, $s > 0$. The following steps are followed:

1) Taylor decomposition of $f$ w.r.t. $x^{\tau} -x$, $\tau = 2^t = \sqrt{\eta}$:
$f(x) = \sum_{i<\tau} f_i(x) (x^{\tau} - x)^i$ with $\deg f_i < \tau$, that is
$f(x) = g(x, x^{\tau}-x)$ with
$$g(x,y) = \sum_{i < \tau} f_i(x) y^i = \sum_{i < \tau, k < \tau} f_{i,k} \ x^k y^i,$$
$f_{i,k} =$ degree-$k$ coefficient of $f_i$. Coefficients of $f_i$ ar written in rows of a square matrix $M$ (row $i$ = $f_i$).

2) Computation of $\tau$ DFT of size $\tau$ on the columns of $M$ to obtain the values of the corresponding polynomials in $W_t + \omega_{\tau j}$; i.e. $\tau$ recursive calls with $s - 1$, coefficients, offset $j$.
Before the DFT, the column $k$ contains the polynomial in $y$
$$P_k = \sum_{i < \tau} f_{i,k} \ y^i$$
in other words, the coefficient at position $(i,k)$ is $f_{i,k}$.
After the DFT, the coefficient $(i,k)$ is
$$\sum_{\ell < \tau} f_{\ell,k} \ \omega_{\tau j + i}^{\ell}.$$
Line $i$ therefore represents the polynomial
$$\sum_{k<\tau} \left (\sum_{\ell < \tau} f_{\ell,k} \: \omega_{\tau j + i}^\ell \right) x^k =
\sum_{\ell < \tau} f_{\ell} \ \omega_{\tau j + i}^{\ell}$$
that is, $f \mod R_i$.

3) Computation of $\tau$ DFT of size $\tau$ on the rows of the matrix, i.e. for all $i < \tau$, the DFT of $f \mod R_i$ on $W_t + \omega_{\eta j + \tau i}$.
On line $i$, we thus recursively call the algorithm with, in addition to the coefficients of the line, the parameters $s-1$ and the offset $\tau j + i$.
As the $i$-th DFT provides the values of $f$ on $W_t + \omega_{\eta j + \tau i}$, if the matrix is in row order in memory, the values are already in the right order for the final return of $\omega_{\eta j + k}$, $k = 0 \ldots \eta - 1$.

### Simplified code

In this prototyping, the `dft` function below performs $2^{\text{logstride}}$ DFTs on as many interleaved polynomials, which simplifies the recursion and limits the number of function calls performed.

~~~C++
void dft(uint64_t* poly, int s, uint64_t j, unsigned int logstride)
{
    if(s==0)
    {
        uint64_t stride = 1uL << logstride;
        uint64_t local_omega = omega(j << 1);
        for(uint64_t i = 0; i < stride; i++)
        {
            poly[i] = poly[i] ^ multiply(poly[i + stride], local_omega);
            poly[i + stride] = poly[i] ^ poly[i + stride];
            //(u = f_0 + f1* \omega_{2.j}, f_0 + f1 * \omega_{2.j+1} = f_0 + f1 *(\omega_{2.j} + 1) = u + f_1)
        }
    }
    else
    {
        unsigned int t = 1uL << (s-1);
        uint64_t eta = 1uL << (2*t);
        uint64_t tau = 1uL << t;
        decompose_taylor(poly, tau, logstride);
        dft(poly, s-1, j, logstride + t);
        for(uint64_t i = 0; i < tau; i++)
        {
            dft(poly+(tau<<logstride), s-1, (j<<t)|i, logstride);
        }
    }
}
~~~

 The main implementation difficulty is to ensure simultaneously a good performance for the multiplication in $\mathbb{F}$ (function `multiply`) and for the evaluation of $j\mapsto \omega_j$ (function `omega`). Computing directly in basis $\beta$ can be tempting (`omega` reduces to the identity in this case), but prevents a very efficient implementation of the product; furthermore, uses of the DFT may require to use a specific basis.
 
 `decompose_taylor` is not shown here but the algorithm is described in [1]. Interstingly, it can be implemented recursively or iteratively, and for a similar implementation effort, the recursive implementation exhibits better memory locality and is faster (in addition to being much simpler).

### Use of a DFT to perform binary polynomial product

In the binary field of size $2^n$, a common basis is generated by a primitive element $\alpha$. If $p$ is the minimal polynomial of $\alpha$ over $\mathbb{F}_2$, the product of two elements of $\mathbb{F}$ in basis $\alpha$ is the product of polynomials over $\mathbb{F}_2$ followed by a modular reduction by $p$. As a consequence, given two elements represented by polynomials of degree $\leq n/2$, their product coincides with the regular polynomial product. This observation can be leveraged as follows to multiply binary polynomials $p$ and $q$ of arbitrary degree:

1) Split $p$ and $q$ into blocks of $n/2$ bits, and map these blocks to coefficients in $\mathbb{F}$, forming two polynomials $\tilde(p)$ and $\tilde(q)$ over $\mathbb{F}$,
1) compute $\tilde{r} = \tilde{p} \times \tilde{q}$ in $\mathbb{F}[x]$.
1) coefficient $i$ of $\tilde{r}$ corresponds to coefficients of degree $i.n/2 \ldots i.n/2 + n - 1$ of $r = p \times q$ (these coefficients overlap and the overlapping parts must be added together).

The product computation at step 2 can be performed by two DFTs, the pointwise product of the results, then an inverse DFT.

Let us detail the correspondence between polynomial products in $\mathbb{F}_2[x]$ and in $\mathbb{F}[x]$. Let $\psi: \mathbb{F}_2[x] \to \mathbb{F}$ which associates $\sum_i u_i x^i$ to $\sum_i u_i \alpha^i$. $\psi$ is bijective when restricted to polynomials of degree $ < n$, $\psi^{-1}$ is the inverse of this restricted map.

A polynomial $p=\sum_i p_i x^i$ si written in "blocks"
$$p=\sum_j c_{p,j} \, x^{j.n/2}\ \text{ with }\ \forall j\,,\ c_{p,j} = \sum_{j.n/2 \leq i < (j+1)n/2} p_i x^{i-j.n/2}$$

Then
$$p\, q = \sum_j \sum_{k+\ell=j} c_{p,k} c_{q,\ell} \, x^{j.n/2} = \sum_j \psi^{-1}\left(\sum_{k+\ell=j} \psi(c_{p,k}) \psi(c_{q,\ell}) \right) \, x^{j.n/2}$$

Write $u_j = \sum_{k+\ell=j} \psi(c_{p,k}) \psi(c_{q,\ell})$. Then $p\, q = \sum_j \psi^{-1}(u_j) x^{j.n/2}$.

Note that the polynomials $\psi^{-1}(u_j)$ are in general of degree $n$, hence they overlap in the sum above.

Compare the product above to the product in $\mathbb{F}[x]$ of $\tilde{p} = \sum_k \psi(c_{p,k}) x^k$ and $\tilde{q} = \sum_{\ell} \psi(c_{q,\ell}) x^{\ell}$:
$$\tilde{p}\, \tilde{q} = \sum_{j} \left(\sum_{k+\ell=j} \psi(c_{p,k}) \psi(c_{q,\ell}) \right) x^j = \sum_j u_j x^j.$$

Therefore this product produces exactly the coefficients needed to compute $p\, q$.

We can derive from this the following method to compute $p \, q$: form the polynomials $\tilde{p}$ and $\tilde{q}$ by forming the coefficients $c(p,j)$ and $c(q,j)$ and, considering them as elements of $\mathbb{F}$ (formally, by computing $\psi(c(p,j))$ and $\psi(c(q,j))$), compute the product $\tilde{p}\, \tilde{q}$. Then use the coefficients of the result, with the proper overlapping, to compute $p\, q$.

As a final remark, note that this algorithm requires to work with the basis $A = \{ 1, \alpha, \alpha^2,\ldots, \alpha^{n-1} \} $, which is distinct from the Cantor basis used for efficient DFT computation. Another incentive to use the basis $A$ is that, with a careful selection of the minimal polynomial of $\alpha$, it enables an efficient implementation of multiplication in $\mathbb{F}$.
