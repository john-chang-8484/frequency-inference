import pstats
from pstats import SortKey

p = pstats.Stats('sim_stats.stats')

# top ten time consuming functions
p.strip_dirs().sort_stats(SortKey.TIME).print_stats(50)

