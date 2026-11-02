import numpy as np, pandas as pd
from scipy.stats import norm
from config import *
log = pd.read_parquet(DATA / "campaign_log.parquet")
seg = pd.read_csv(TAB / "customer_segments.csv")[["customer_id","persona"]]
d = log.merge(seg, on="customer_id")
def lift(g):
    a, b = g[g.variant=="persona"].converted, g[g.variant=="generic"].converted
    pa, pb = a.mean(), b.mean(); se = np.sqrt(pa*(1-pa)/len(a) + pb*(1-pb)/len(b))
    pool = g.converted.mean(); z = (pa-pb)/np.sqrt(pool*(1-pool)*(1/len(a)+1/len(b)))
    return pd.Series({"n": len(g), "conv_generic": pb, "conv_persona": pa,
        "rel_lift_pct": 100*(pa/pb-1) if pb else np.nan, "p_value": 2*(1-norm.cdf(abs(z))),
        "abs_diff_ci_lo": pa-pb-1.96*se, "abs_diff_ci_hi": pa-pb+1.96*se})
res = d.groupby("persona").apply(lift, include_groups=False)
tot = lift(d).rename("ALL"); res = pd.concat([res, tot.to_frame().T]).round(4)
print(res.to_string()); res.to_csv(TAB / "campaign_results.csv")
