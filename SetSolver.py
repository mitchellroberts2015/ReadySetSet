def set_solver(input_cards):
    rv = []

    for i in range(len(input_cards)):
        for j in range(i+1, len(input_cards)):
            c1 = input_cards[i]
            c2 = input_cards[j]
            print('c1', c1)
            print('c2', c2)
            matches = []
            for c1_k, c2_k in zip(c1, c2):
                # print(c1_k, c2_k, c1_k if c1_k == c2_k else ~(c1_k^c2_k)%3)
                # print('lower', lower)
                # print('higher', higher)
                A = c1_k&0x2
                B = c1_k&0x1
                C = c2_k&0x2
                D = c2_k&0x1
                lower = ~C & ~A
                higher = D & C
                other = 0 if ((c1_k==1 and c2_k==2) or (c1_k==2 and c2_k==1)) else 1 if ((c1_k==0 and c2_k==2) or (c1_k==2 and c2_k==0)) else 2
                matches.append(c1_k if c1_k == c2_k else other)
            print('c3', matches)
            print(matches in input_cards and set((i, j, input_cards.index(matches))) not in rv)
            if matches in input_cards and set((i, j, input_cards.index(matches))) not in rv:
                print('adding', set((i, j, input_cards.index(matches))))
                rv.append(set((i, j, input_cards.index(matches))))
    return rv

print(set_solver([[1, 1, 1, 2], [1, 1, 1, 1], [1,1,1,0], [2, 0, 1, 0]]))