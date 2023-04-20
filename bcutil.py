import pandas as pd

def cor2rel(C, threshold=0.5):
    nvar = C.shape[0]
    rel = []
    for i in range(0,nvar-1):
        for j in range(i+1, nvar):
            a = C.columns[i]
            b = C.columns[j]
            r = C.iloc[i, j] 
            if r > threshold:
                rel.append((a,b,1))
            else:
                pass
           #rel.append((a,b,0)) 
    rel_df = pd.DataFrame(rel)
    rel_df.columns = ["Source", "Target", "Value"]
    return rel_df


def cor2rel_part(C, numcol, threshold=0.5):
    nvar = C.shape[0]
    rel = []
    for i in range(0, numcol):
        for j in range(i+1, nvar):
            a = C.columns[i]
            b = C.columns[j]
            r = C.iloc[i, j] 
            if r > threshold:
                rel.append((a,b,1))
            else:
                pass
           #rel.append((a,b,0)) 
    rel_df = pd.DataFrame(rel)
    rel_df.columns = ["Source", "Target", "Value"]
    return rel_df
