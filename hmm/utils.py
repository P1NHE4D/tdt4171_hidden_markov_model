

def print_probabilities(prob_sequence, start_index=0):
    """
    Prints the given probability sequence

    :param prob_sequence: Probability sequence for all states j in a given time period t (array<t,j>)
    :param start_index: Start index of t (Default: 0)
    """

    header = " t  "
    for i in range(1, prob_sequence.shape[1] + 1):
        header += "|  x_{}  ".format(i)
    line = "-" * len(header)
    print(header)
    print(line)

    for timestep, ps in enumerate(prob_sequence):
        row = " {}  ".format(start_index + timestep)
        for p in ps:
            row += "| {:.3f} ".format(p)
        print(row)
    print()
