# approximation_of_pi
The goal of this HW is to study the speed performance (i.e. runtime) between 
the following three different algorithm styles:
 1) Vectorized
 2) Recursive (without memoization)
 3) Recursive with memoization
In other words, do the same task, in three different ways. Find which way is 
faster.

Suppose the task at hand is to evaluate accuracy of the approximated value of 
pi using four different infinite series or product equations:
 1) Wallis' equation (w)
 2) The Gregory-Leibniz equation (gl)
 3) Euler's Basel equation (eb)
 4) Viete's equation (v)
All four equations above converge to the actual value of pi if infinite number 
of terms are used. However, infinite number of terms is infeasible. 
Nonetheless, the accuracy of the approximated value improves as more terms are 
used in the equation. The goal is to plot approximated pi value as a function 
of k, where k is the number of terms used. Repeat this for each of the four 
equations.

In summary, the purpose of this script is to generate the following figures.

Fig.1: Vectorized style (abbreviated as "vec")
 Compare results between w vs. gl vs. eb (Viete's cannot be vectorized)
 subplot(211) plot of approximated pi value vs. k
 subplot(212) plot of absolute error between the approximation and the actual 
              pi value vs. k
 
Fig.2: Recursive style (abbreviated as "rec")
 Compare results between w vs. gl vs. eb vs. v
 subplot(211) and (212) same as those of Fig. 1
 
Fig.3: Recursive with memoization style (abbreviated as "rm")
 Compare results between w vs. gl vs. eb vs. v
 subplot(211) and (212) same as those of Fig. 1       

Fig.1-3 should give similar looking plots. The results should match.

Fig.4: Compare runtime between all methods (all combinations of pi equation and 
algorith style, i.e.    w-vec, gl-vec, eb-vec
                        w-rec, gl-rec, eb-rec, v-rec
                        w-rm , gl-rm , eb-rm , v-rm )
