def load_groceries_data(file):
    """Load the groceries file, read the data and returns a list of baskets
    Parameters
    ----------
    file : str
        Groceries file path
    Returns
    -------
    list
        a list of baskets
    """
    baskets = []
    with open(file, 'r', encoding='utf-8') as groceries_file:
        lines = groceries_file.read().splitlines()
        baskets = [l.split(',') for l in lines]
    print('Loaded %d baskets' % len(baskets))
    return baskets