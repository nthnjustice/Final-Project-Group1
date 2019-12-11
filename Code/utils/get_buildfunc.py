# crosswalk for buildings dataset - maps building type labels from Dutch to English


# https://github.com/SPINLab/geometry-learning/blob/develop/prep/preprocess-buildings.py
def get_buildfunc(building):
    if building == 'bijeenkomstfunctie':
        return 'gatherings'
    elif building == 'industriefunctie':
        return 'industrial'
    elif building == 'logiesfunctie':
        return 'lodging'
    elif building == 'woonfunctie':
        return 'habitation'
    elif building == 'winkelfunctie':
        return 'shopping'
    elif building == 'kantoorfunctie':
        return 'office'
    elif building == 'gezondheidszorgfunctie':
        return 'health care'
    elif building == 'onderwijsfunctie':
        return 'educational'
    elif building == 'sportfunctie':
        return 'sports'
    else:
        return 'unk'
