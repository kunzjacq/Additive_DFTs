# Faster Taylor decompositions: merging two levels, and a row kernel

Status: **applied to the fused variant of the product** (`libfft/mg_fused.cpp`), through the functions of
`include/mg_taylor_opt.h`; the plain variant (`libfft/mg.cpp`) still uses the original functions of
`include/mg_templates.h`. See [algo_fft_en.md](algo_fft_en.md), section "Taylor decomposition", for the
algorithm itself (notations: $\tau = 2^t$, $n$ the size of the polynomial, $m = n/2$, $\delta = m/\tau$,
$Y = X^\tau + X$).

This document describes the optimizations, how they were checked, the measured gains, the experiments that
did not help, and the possible improvements.

## Motivation

The Taylor decompositions (`mg_decompose_taylor_recursive` and `mg_decompose_taylor_reverse_recursive` in
`include/mg_templates.h`) use no multiplication, but they take a significant share of the time of a product.
Measured on a Ryzen 9 5950X by replacing them with no-ops (the results are then wrong, but the running time
of the rest does not depend on the data), in-place balanced products of $2^u$ bits:

| $u$ | plain variant | fused variant (before the optimizations) |
|---|---|---|
| 10 | 14% | 0% (base case, no DFT) |
| 12 | 24% | 0% |
| 14 | 26% | 0% |
| 16 | 27% | 0% |
| 18 | 29% | 17% |
| 20 | 28% | 16% |
| 22 | 32% | 25% |
| 24 | 33% | 30% |

A `perf` profile (with the Taylor functions forbidden to inline) agrees at large sizes, and shows that the cost
is spread over about 30 instantiations of the templates (one per pair (`logstride`, `t`)), the largest one
being 3.5% of the total. About two thirds of the time are in the forward decompositions, one third in the
reverse ones. The small values of $\tau$ are the main contributors. Time that would be saved if the steps for
a given $\tau$ were free (upper bounds, measured by replacing them with no-ops; noise about 1 to 2%):

| $u$ | variant | $\tau = 2$ | $\tau = 4$ | $\tau = 16$ | all $\tau \leq 16$ |
|---|---|---|---|---|---|
| 14 | plain | 0% | 17% | 0% | 25% |
| 18 | plain | 3% | 14% | 3% | 28% |
| 20 | plain | 2% | 13% | 4% | 27% |
| 24 | plain | 4% | 12% | 5% | 27% |
| 18 | fused | 1% | 1% | 8% | 16% |
| 20 | fused | 1% | 1% | 9% | 16% |
| 24 | fused | 3% | 4% | 9% | 19% |

($\tau = 4$ is the Taylor decomposition of rows of 8 and 16 words and of the corresponding interleaved
columns; the fused variant skips most of the rows of 16 words, hence its low value.)

Each level of the recursion is a pass over the array. The first idea is to do two levels at once, to reduce the
number of memory operations; the second is to do the narrowest decompositions in registers.

## 1. Merging two levels

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

Written as 8 separate loops, the merged step does the same number of additions and the same number of
memory operations as the original, and gives no gain. The implementation fuses steps 5 and 6, which both
modify $A$: one loop does $A[p] \mathrel{\hat{}}= C[p-2g] + B[p-g]$ for $p \geq 2g$ (and
$A[p] \mathrel{\hat{}}= B[p-g]$ for $g \leq p < 2g$). This saves one read-modify-write of a quarter of the
array out of four on a pair of levels.

The recursion is: if there are at least two levels left ($\text{logsize} \geq t+2$), do the merged step and
recurse on the four quarters with $\text{logsize} - 2$; otherwise do the usual one-level step.

## 2. The row kernel for $\tau = 4$

For $t = 2$ and $\text{logstride} = 0$ (a single polynomial of 8 or 16 words), the whole decomposition fits in
AVX2 registers (4 words per register), and is done without loops (`taylor_row8_forward`,
`taylor_row16_forward` and their reverse versions in `include/mg_taylor_opt.h`).

- For 16 words ($n = 16$, two levels, $g = 1$, $q = 4$), the quarters $A, B, C, D$ are one 256-bit register each
  and the 8 steps above become register operations: the shifts by 1 and 2 words are a permute and a blend
  (`taylor_shift_up`: $(x_0, x_1, x_2, x_3) \mapsto (0, x_0, x_1, x_2)$) and moves of a 128-bit half
  (`taylor_hi_to_lo`, `taylor_lo_to_hi`), the steps 4 and 7 add the word 3 to the word 0
  (`taylor_lane3_to_lane0`).
- For 8 words (one level, $m = 4$, $\delta = 1$), the loop does $\text{hi}_0 \mathrel{\hat{}}= \text{hi}_3$, then
  $\text{lo}_j \mathrel{\hat{}}= \text{hi}_{j-1}$, $j \geq 1$: one blend, one shift, two additions.

Loads and stores are done once per row, and every word is written once. The reverse kernels apply the same
operations in reverse order. They are only used for words of 64 bits; the other cases (interleaved
polynomials, 32-bit words) use the generic code with the merged step.

## Checks

The new functions (`mg_decompose_taylor_opt` and `mg_decompose_taylor_reverse_opt`) are compared to the
original ones by `taylor_test` in `product_test.cpp`, on random arrays, for $t = 1 \ldots 5$, various strides,
all sizes allowed (`t < logsize <= 2t`), 64-bit and 32-bit words, forward and reverse: no difference.
Complete products give the same results: `degree_test` (30254 cases) and the comparison with gf2x pass, for both
variants (the plain variant does not use the new functions).

## Results

Throughput of whole in-place products, new over original (best of 3 runs of 2 seconds, same process;
prototypes measured before the code was added to the tree):

| $u$ | plain, merge | plain, row kernel | plain, both | fused, both |
|---|---|---|---|---|
| 12 | 1.04 | 1.08 | 1.17 | 1.00 |
| 16 | 1.06 | 1.04 | 1.11 | 1.00 |
| 18 | 1.06 | 1.04 | 1.11 | 1.01 |
| 20 | 1.08 | 1.05 | 1.13 | 1.01 |
| 22 | 1.04 | 1.03 | 1.10 | 1.03 |
| 24 | 1.06 | 1.03 | 1.09 | 1.05 |

(For the fused variant, there is no gain for $u \leq 16$ as it does not use the DFT there. At $u = 30$,
one product of the fused variant takes 1.366 s with the original functions, and 1.283 s with both
optimizations (1.06 times faster); the plain variant takes 1.552 s and 1.444 s (1.07 times faster).)
In the tree (fused variant only; old and new code built with the same flags, minimum of 3 alternating runs), the
products are 1.06 times faster at $u = 18$, 1.05 at $u = 20$ and 22, 1.03 at $u = 24$, 1.04 at $u = 26$ and 1.05
at $u = 28$ and 30.
The plain variant is not modified by the tree: the gains in its column are those of the prototype, which could
be obtained by using the new functions in `mg.cpp` too.

The gains are modest because the Taylor decompositions are about 30% of the time at large sizes, and the
optimizations remove part of them. The row kernel only concerns the rows ($\text{logstride} = 0$), the
interleaved columns with $\tau = 4$ remain, and the fused variant skips most of the rows of 16 words.

## Experiments without gain

- **Fixed-size kernels for $\tau = 16$** (rows of 32, 64, 128, 256 words, $\text{logstride} = 0$): the two-level
  network and the one-level step written with all lengths and offsets as compile-time constants (so that
  the compiler unrolls everything, without any run-time check). Verified identical to the original, but on top
  of the row kernel and the merge, no measurable gain (between $-2\%$ and $+1.5\%$). The rows are already
  small and in L1; the remaining time of $\tau = 16$ is partly in the interleaved columns
  ($\text{logstride} = 8$, arrays of 64 to 512 KB), where the loops are long and the cost is that of the
  passes on memory, not of the loop overhead.
- **Templating on `logstride` and `t`:** replacing the templates by run-time parameters makes whole
  products 4 to 9% slower at large sizes (templating one of the two parameters only gives half of the
  loss), and shrinks the code of `mg.cpp` from 167 KB to 130 KB. Probably the compiler can vectorize the
  loops without peeling when it knows that all offsets and lengths are multiples of $2^{\text{logstride}}$.
  The templates are kept.
- **Skipping intermediate writes in general.** Observation: in one level, the words $[m+\delta, 2m)$ are only
  read and never written, and the targets in $[2\delta, m)$ are only written (once per level). This does not
  extend across levels: most of $[m+\delta, 2m)$ is written by the next levels, and all of $[2\delta, m)$ is
  read by them (it becomes the lower half whose upper half contains the sources of the next level). Only
  $\tau$ words (the word 0 and the top $\tau - 1$ words) are never written. A symbolic simulation shows that
  29 to 41% of the writes are overwritten before being read ("dead writes"), but only 1 to 10% of the
  writes are dead and have a source that is unchanged until the next write, i.e. can be folded into it
  without extra work (the fusion of steps 5 and 6 is one such case). Computing each final word from the
  original values would need 1.5 to 8 times more additions than the recursion (42623 against 12288 for
  $n = 4096$, $\tau = 64$), as the recursion reuses partial sums.

## Possible improvements

1. **Fuse the other passes of the merged step.** Steps 2, 3 and 4 all modify $B$, and steps 1 and 8 both
   modify $C$ (and step 7 $D$); they could be done in one loop per quarter, as steps 5 and 6 were. The
   beginning and the end of the ranges (length $g$ or $2g$) need separate loops. The expected gain is another
   few percent of the time of a product. Note that step 1 must still be done before step 5 reads $C$, so
   fusing steps 1 and 8 requires reading $C$ and the end of $D$ explicitly in the loop of $A$.
2. **Merge more levels.** The same derivation can be done for three levels at a time (eight octants, shifts
   $4g'$, $2g'$, $g'$): the number of passes on the array is divided by three instead of two. The number of
   additions is unchanged. This should pay off when the array does not fit in the L2 cache, in particular
   for the interleaved columns with $\tau = 16$ and $\tau = 256$, where the fixed-size kernels do not help.
3. **Kernels for the interleaved case and for $\tau = 2$.** The row kernel only handles $\text{logstride} = 0$.
   For interleaved polynomials ($\text{logstride} > 0$, blocks of $2^{\text{logstride}}$ words) the same 8
   steps can be written on blocks, with every destination block written once; the extra additions of a
   fully direct evaluation (1.4 times for $n = 16$) may be worth it for $\tau = 4$.
4. **Use the new functions in the plain variant too.** The plain variant (`mg.cpp`) was kept unchanged; the
   prototype gains 9 to 17% there.
5. **Fuse the decomposition with the stage that follows.** After the decomposition, the column DFTs read the
   array again from the beginning; for sizes that do not fit in cache, doing the decomposition by blocks
   and starting the next stage on a block as soon as it is complete would save a pass over memory.
6. **Skip the zero padding.** The fast code pads the polynomials to a power of 2 with zeros and the loops
   run on all the words. The generic version in `include/decompose_taylor.h` only handles the existing
   coefficients. For balanced products the gain is at most a factor of 2 on the first levels of the
   operands' decompositions, and zero for powers of 2.
7. **Check the code generated for the loops.** The original loops of the one-level step go downwards, and
   the compiler must check for overlaps between the source and the destination (they are not disjoint),
   so the narrow loops end up scalar. The loops of the merged step are between disjoint ranges declared
   `restrict`, and can be vectorized without any run-time check; explicit intrinsics with unaligned 256-bit
   loads would make this certain.
