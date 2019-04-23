def set_solver(input_cards):
    rv = []
    input_cards = [[int(el) for el in c] for c in input_cards]

    for i in range(len(input_cards)):
        for j in range(i+1, len(input_cards)):
            c1 = input_cards[i]
            c2 = input_cards[j]
            matches = []
            # print('c1', c1)
            # print('c2', c2)
            for c1_k, c2_k in zip(c1, c2):
                other = 0 if ((c1_k==1 and c2_k==2) or (c1_k==2 and c2_k==1)) else 1 if ((c1_k==0 and c2_k==2) or (c1_k==2 and c2_k==0)) else 2
                matches.append(c1_k if c1_k == c2_k else other)
            # print('c3', matches)
            if matches in input_cards and set((i, j, input_cards.index(matches))) not in rv:
                rv.append(set((i, j, input_cards.index(matches))))
    return rv
