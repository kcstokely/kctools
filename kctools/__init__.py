'''kctools'''

from .kctools import lsdashr
from .kctools import readlines
from .kctools import lpad
from .kctools import sbool
from .kctools import html_strip
from .kctools import rem_punc
from .kctools import rep_punc
from .kctools import tnow
from .kctools import endify
from .kctools import humanify
from .kctools import only_one
from .kctools import split_into_rows
from .kctools import split_into_chunks
from .kctools import flatten
from .kctools import gram_getter
from .kctools import where_in_thing
from .kctools import autovivify
from .kctools import mortify
from .kctools import dict_update
from .kctools import setup_logger
from .kctools import coalesce
from .kctools import mround

from .kctools import normalize
from .kctools import is_diagonal
from .kctools import kpow
from .kctools import kcos
from .kctools import bin_entropy
from .kctools import mod_entropy
from .kctools import lower_conf_bound
from .kctools import upper_conf_bound
from .kctools import conf_bounds
from .kctools import pargsort
from .kctools import psort
from .kctools import rchoice
from .kctools import rename_dup_df_cols
from .kctools import make_heatmap

from .classes.dicts import adict, odict
from .classes.cuke  import Cuke
from .classes.lock  import Lock
from .classes.map   import Map
from .classes.sub   import N
from .classes.reg   import Register
from .classes.vec   import Vector, VecSet
