# Merging two levels of the Taylor decomposition: an experiment

Status: **prototype, not applied to the code**. This document describes the approach, how it was checked,
the measured gain, and the possible improvements. See [algo_fft_en.md](algo_fft_en.md), section "Taylor
decomposition", for the algorithm itself (notations: $\tau = 2^t$, $n$ the size of the polynomial,
$m = n/2$, $\delta = m/\tau$, $Y = X^\tau + X$).

## Motivation

The Taylor decompositions (`mg_decompose_taylor_recursive` and `mg_decompose_taylor_reverse_recursive` in
`include/mg_templates.h`) use no multiplication, but they take a significant share of the time of a product.
Measured on a Ryzen 9 5950X by replacing them with no-ops (the results are then wrong, but the running time
of the rest does not depend on the data), in-place balanced products of $2^u$ bits:

| $u$ | plain variant | fused variant |
|---|---|---|
| 10 | 14% | 0% (base case, no DFT) |
| 12 | 24% | 0% |
| 14 | 26% | 0% |
| 16 | 27% | 0% |
| 18 | 29% | 17% |
| 20 | 28% | 16% |
| 22 | 32% | 25% |
| 24 | 33% | 30% |

A `perf` profile (with the Taylor functions forbidden to inline) agrees at large sizes (29% and 33% at
$u = 24$), and shows that the cost is spread over about 30 instantiations of the templates (one per pair
(`logstride`, `t`)), the largest one being 3.5% of the total (rows of 16 words, `t = 2`, `logstride = 0`).
About two thirds of the time are in the forward decompositions, one third in the reverse ones.

Each level of the recursion is a pass over the array. The idea tested here is to do two levels at once, to
reduce the number of memory operations.

## The merged step

Take the one-level step of the recursion on $n$ words: a loop of $m = n/2$ additions, then the same step on
each half. Doing two levels means performing the level-1 step, then the level-2 step on each of the two
halves. Let $q = n/4$ and let $A, B, C, D$ be the four quarters of the array (of $q$ words each). Let
$g = \delta/2 = n/(4\tau)$: the shift of the level-1 step is $\delta = 2g$, that of the level-2 step is $g$,
and $q = \tau g$. We assume $\tau \geq 4$ ($t \geq 2$, so that $q \geq 4g$; for $t = 1$ the size is at most 4
and there are no two levels to merge). All sizes and offsets are multiplied by $2^{\text{logstride}}$ in
the code for interleaved polynomials; this does not change the formulas.

With $X[a..b)$ the words $a, \ldots, b-1$ of a quarter $X$, and $+$ the addition (XOR):

- Level 1 adds the high half $(C|D)$ shifted by $\delta = 2g$ words to the array. The part of the sum that
  falls beyond the low half modifies the beginning of $C$ before it is read again:
  $C'[0..2g) = C[0..2g) + D[q-2g..q)$ ($C' = C$ elsewhere). Then the low half becomes
  $v_1[p] = A[p] + C'[p-2g]$ for $p \geq 2g$ ($v_1 = A$ otherwise) and $v_2[p] = B[p] + C'[q-2g+p]$ for
  $p < 2g$, $v_2[p] = B[p] + D[p-2g]$ for $p \geq 2g$. The high half is $u = (C' | D)$.
- Level 2 acts in the same way on $v = (v_1 | v_2)$ and $u = (u_1 | u_2) = (C' | D)$, with shift $g$:
  $v_2'[0..g) = v_2[0..g) + v_2[q-g..q)$, $v_1'[p] = v_1[p] + v_2'[p-g]$ for $p \geq g$;
  $D'[0..g) = D[0..g) + D[q-g..q)$, $C''[p] = C'[p] + D'[p-g]$ for $p \geq g$.

The result is the array $(v_1' | v_2' | C'' | D')$, followed by the recursion on its four quarters. Writing
this in place, with the order of the operations chosen so that every value is read before it is
overwritten, gives 8 additions of ranges ($\mathrm{dst}[\ldots] \mathrel{\hat{}}= \mathrm{src}[\ldots]$),
all between disjoint ranges:

| step | operation | length |
|---|---|---|
| 1 | $C[0..2g) \mathrel{\hat{}}= D[q-2g..q)$ | $2g$ |
| 2 | $B[0..2g) \mathrel{\hat{}}= C[q-2g..q)$ | $2g$ |
| 3 | $B[2g..q) \mathrel{\hat{}}= D[0..q-2g)$ | $q-2g$ |
| 4 | $B[0..g) \mathrel{\hat{}}= B[q-g..q)$ | $g$ |
| 5 | $A[2g..q) \mathrel{\hat{}}= C[0..q-2g)$ | $q-2g$ |
| 6 | $A[g..q) \mathrel{\hat{}}= B[0..q-g)$ | $q-g$ |
| 7 | $D[0..g) \mathrel{\hat{}}= D[q-g..q)$ | $g$ |
| 8 | $C[g..q) \mathrel{\hat{}}= D[0..q-g)$ | $q-g$ |

The dependencies that fix the order: step 1 reads the end of $D$ before step 7 changes its beginning; steps
2 and 3 read $C$ (beyond its modified beginning, since $q - 2g \geq 2g$) and the beginning of $D$ before
steps 7 and 8; step 4 needs the final values of the end of $B$; steps 5 and 6 use $C$ after step 1 and $B$
after steps 2 to 4, and are done before step 8 changes $C$; step 8 uses $D'$, after step 7.

The total number of additions is $4q = n$, the same as with two levels of the original recursion. The
reverse transform recurses on the four quarters first, then performs the same 8 additions in reverse order
(each addition is its own inverse, and its source is not modified by it).

The experiment replaced the recursion by: if there are at least two levels left ($\text{logsize} \geq t+2$),
do the merged step and recurse on the four quarters with $\text{logsize} - 2$; otherwise do the usual one-level step.

## Check

The merged forward and reverse functions were compared to the original ones on random arrays for
$t = 1 \ldots 5$, strides $2^0 \ldots 2^8$ and all sizes allowed (by `t < logsize <= 2t`): no difference.
Complete products, with the new functions in place of the originals, give the same results: `degree_test`
(30254 cases) and the comparison with gf2x from $2^8$ to $2^{22}$ bits pass, for both variants.

## Result

Throughput of whole in-place products (best of 3 runs of 2 seconds, same process, original over new):

| $u$ | plain, 8 separate range additions | plain, steps 5 and 6 fused | fused variant, steps 5 and 6 fused |
|---|---|---|---|
| 12 | 0.99 | 1.04 | 1.00 |
| 16 | 0.98 | 1.06 | 1.00 |
| 18 | 0.99 | 1.06 | 1.02 |
| 20 | 0.99 | 1.08 | 1.02 |
| 22 | 0.99 | 1.04 | 1.03 |
| 24 | 1.00 | 1.06 | 1.04 |

(The gain is 1.0 for the fused variant for $u \leq 16$, as it does not use the DFT there.)

- Written as 8 separate loops, the merged step does the same number of additions and the same number of
  memory operations as the original: **no gain** (the differences are within measurement noise).
- The only fusion done in the second experiment is that of steps 5 and 6, which both modify the first quarter
  $A$: one loop does $A[p] \mathrel{\hat{}}= C[p-2g] + B[p-g]$ for $p \geq 2g$ (and $A[p] \mathrel{\hat{}}= B[p-g]$ for
  $g \leq p < 2g$). This saves one read-modify-write of a quarter of the array out of four on a pair of levels,
  and gives **4 to 8% on the plain variant and 2 to 4% on the fused variant** for $u \geq 18$.

The gain is small because the Taylor decompositions are about 30% of the time at these sizes, and one pass
out of four of one pair of levels is a small part of this. The code of the prototype is given below.

## Prototype code

~~~C++
// dst[i] ^= src[i], i < len, for disjoint ranges
template<class T>
static inline void taylor_xor_range(T* restr dst, const T* restr src, uint64_t len)
{ for(uint64_t i = 0; i < len; i++) dst[i] ^= src[i]; }

// dst[i] ^= s1[i] ^ s2[i]
template<class T>
static inline void taylor_xor2(T* restr dst, const T* restr s1, const T* restr s2, uint64_t len)
{ for(uint64_t i = 0; i < len; i++) dst[i] ^= s1[i] ^ s2[i]; }

// in mg_decompose_taylor_recursive<logstride, t, T>(logsize, p), before the one-level step:
if constexpr(t >= 2)
{
  if(logsize >= t + 2)
  {
    const uint64_t n_s = 1uLL << (logstride + logsize);
    const uint64_t q_s = n_s >> 2;
    const uint64_t g_s = 1uLL << (logstride + logsize - 2 - t);
    T* restr A = p; T* restr B = p + q_s; T* restr C = p + 2 * q_s; T* restr D = p + 3 * q_s;
    taylor_xor_range(C, D + q_s - 2 * g_s, 2 * g_s);                      // step 1
    taylor_xor_range(B, C + q_s - 2 * g_s, 2 * g_s);                      // step 2
    taylor_xor_range(B + 2 * g_s, D, q_s - 2 * g_s);                      // step 3
    taylor_xor_range(B, B + q_s - g_s, g_s);                              // step 4
    taylor_xor_range(A + g_s, B, g_s);                                    // steps 5 and 6 (fused)
    taylor_xor2(A + 2 * g_s, C, B + g_s, q_s - 2 * g_s);
    taylor_xor_range(D, D + q_s - g_s, g_s);                              // step 7
    taylor_xor_range(C + g_s, D, q_s - g_s);                              // step 8
    if(logsize > t + 2) { /* recursive call on A, B, C, D with logsize - 2 */ }
    return;
  }
}
// ... usual one-level step
~~~

For the reverse function, after the recursive calls on the four quarters, the same operations are done in the
order 8, 7, (6 and 5 fused: `taylor_xor2(A + 2*g_s, C, B + g_s, ...)` and `taylor_xor_range(A + g_s, B, g_s)`),
4, 3, 2, 1.

## Possible improvements

1. **Fuse the other passes of the merged step.** Steps 2, 3 and 4 all modify $B$, and steps 1 and 8 both
   modify $C$ (and step 7 $D$); they could be done in one loop per quarter, as steps 5 and 6 were. The
   beginning and the end of the ranges (length $g$ or $2g$) need separate loops. The expected gain is another
   few percent of the time of a product. Note that step 1 must still be done before step 5 reads $C$, so
   fusing steps 1 and 8 requires reading $C$ and the end of $D$ explicitly in the loop of $A$.
2. **Merge more levels.** The same derivation can be done for three levels at a time (eight octants, shifts
   $4g'$, $2g'$, $g'$): the number of passes on the array is divided by three instead of two. The number of
   additions is unchanged. This should pay off when the array does not fit in the L2 cache.
3. **Specialized in-register code for the narrowest levels.** The largest single contributor is the
   decomposition of rows of 16 words ($t = 2$, `logstride = 0`, $n = 16$, two levels, $g = 1$, $q = 4$): the
   whole row, 16 words, is 4 256-bit vectors and the four quarters are one vector each. The transform is
   then a fixed sequence of 8 range additions with shifts by 1 and 2 words, which can be done with a few
   permutations and XORs in registers, like the kernel for rows of 4 words in the DFT. The other narrow
   instantiations (`t = 2`, various strides, and `t = 1`) could be handled in the same way. This code
   should also be used by the reverse transform (the inverse is the same sequence in reverse order).
4. **Fuse the decomposition with the stage that follows.** After the decomposition, the column DFTs read the
   array again from the beginning; for sizes that do not fit in cache, doing the decomposition by blocks
   and starting the next stage on a block as soon as it is complete would save a pass over memory.
5. **Skip the zero padding.** The fast code pads the polynomials to a power of 2 with zeros and the loops
   run on all the words. The generic version in `include/decompose_taylor.h` only handles the existing
   coefficients. For balanced products the gain is at most a factor of 2 on the first levels of the
   operands' decompositions, and zero for powers of 2.
6. **Check the code generated for the loops.** The original loops of the one-level step go downwards, and
   the compiler must check for overlaps between the source and the destination (they are not disjoint),
   so the narrow loops end up scalar. The loops of the merged step are between disjoint ranges declared
   `restrict`, and can be vectorized without any run-time check; explicit intrinsics with unaligned 256-bit
   loads would make this certain.
7. **Which variant gets it.** The Taylor functions are shared by `mg.cpp` and `mg_fused.cpp` through
   `mg_templates.h`. If the plain implementation must stay unchanged, the merged functions should be added
   under another name and used only by `mg_fused.cpp`.
