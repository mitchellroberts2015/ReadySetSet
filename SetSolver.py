def set_solver(input_cards):
    rv = []

    for i in range(len(input_cards)):
        for j in range(i+1, len(input_cards)):
            c1 = input_cards[i]
            c2 = input_cards[j]
            matches = []
            for c1_k, c2_k in zip(c1, c2):
                matches.append(c1_k if c1_k == c2_k else (c1_k^c2_k)%3)
            if matches in input_cards and set((i, j, input_cards.index(matches))) not in rv:
                rv.append(set((i, j, input_cards.index(matches))))
    return rv

print(set_solver([[1, 1, 1, 1], [1, 2, 2, 2], [1,3,3,2], [2, 3, 1, 0]]))