import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    def dot_product(a,b):
        res =0
        for i in range(len(a)):
            res += a[i]*b[i]
        return res
    if len(vectors)==1:
        return []
        
    ortonormal = []
    for idx, vec in enumerate(vectors):
        x=0
        temp = vectors[idx]
        for i in range(idx):
            numerator_dot = dot_product(vectors[idx],ortonormal[i])
            denominator_dot = dot_product(ortonormal[i],ortonormal[i])
            frac = numerator_dot/(denominator_dot+tol)
            mod=[]
            for num  in ortonormal[i]:
                mod.append(frac * num)
            for j in range(len(vectors[idx])):
                temp[j] -= mod[j]
        final = (temp)/np.sqrt(dot_product(temp,temp))
        sq_norm = dot_product(temp, temp)
        if sq_norm > tol:
            final = np.array(temp) / np.sqrt(sq_norm)
            ortonormal.append(final)
        
    return ortonormal