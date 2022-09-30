## Ranking of multiple submissions and metrics

The ranking scheme is in priniciple based on the ranking scheme of the Medical Decathlon. We rank methods using statistically significantly different results. For each metric applied in a task, methods are compared against each other (Wilcoxon signed rank test with p<0.05, see details in code), ranked based on the number of ”won” comparisons and finally mapped to a numerical metric rank score between 0.1 and 1 (with possible score sharing). 

A task rank score is then obtained as the geometric mean of individual metric rank scores. All methods for which no metric is available (not submitted to the task, no Docker container submitted) share the lowest possible metric rank score of 0.1.
