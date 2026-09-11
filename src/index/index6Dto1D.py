def index(ind1, ind2, ind3, ind4, ind5, ind6, size1, size2, size3, size4, size5, size6):
    indOut = ind6 + size6 * ind5 + size6 * size5 * ind4 + size6 * size5 * size4 * ind3 + size6 * size5 * size4 * size3 * ind2 + size6 * size5 * size4 * size3 * size2 * ind1
    return indOut
