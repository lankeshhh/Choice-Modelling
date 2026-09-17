# `bakery.txt`

Vendored unmodified from
[arbenson/discrete-subset-choice](https://github.com/arbenson/discrete-subset-choice)
(commit at time of vendoring: see `git log` for the commit this file was
added in this repo). 75,000 lines, 50 unique items; each line is a
subset selection (a shopping basket) from a bakery's universal choice set.
No item names, prices, categories, or other metadata are included in the
source data -- only integer item IDs.

Please cite the accompanying paper if you use this data:

> Austin R. Benson, Ravi Kumar, and Andrew Tomkins. A Discrete Choice
> Model for Subset Selection. Proceedings of the eleventh ACM
> International Conference on Web Search and Data Mining (WSDM), 2018.

See `src/data/bakery.py` for how this repo converts the raw
subset-selection format into single-choice training data, and
`decisions.md` for the reasoning behind that conversion and its
limitations.
