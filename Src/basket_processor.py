class BasketProcessor:

    def __init__(self):
        pass

    def load_groceries_data(self, file):
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

    def find_common_items(self, vocabulary, unique_items):
        #TODO: This is comparing exact values (Use similarities to include similar e.g. bottled water = water, whipped/sour cream = whiped sour cream) Use any simmilarity to improve this
        """Collect the items that are present in the vocabulary of ingredients from the receipes dataset
        Parameters
        ----------
        vocabulary : list
            vocabulary of final ingredients from the receipes dataset
        unique_items: list
            list of unique items in the baskets
        Returns
        -------
        list
            a tuple of discarded items and kept items
        """
        discarded_indexes = [item not in vocabulary for item in unique_items]
        self.discarded, self.kept = [], []
        for i, discard in enumerate(discarded_indexes):
            t = self.discarded if discard == True else self.kept
            t.append(unique_items[i])
        print('Intersection between unique items and ingredients: %d' % len(self.kept))

    def get_filtered_baskets(self, baskets):
        """filter baskets keeping only kept_items items
        Parameters
        ----------
        baskets : list
            list of baskets
        kept_items: list
            items to keep in the baskets
        Returns
        -------
        list
            a list of the resulting baskets
        """
        filtered_baskets = []
        for basket in baskets:
            filtered_basket = []
            for item in basket:
                if item in self.kept:
                    filtered_basket.append(item)
            if filtered_basket:
                filtered_baskets.append(filtered_basket)
        print('Filtered baskets lenght %d out of %d' % (len(filtered_baskets), len(baskets)))
        return filtered_baskets