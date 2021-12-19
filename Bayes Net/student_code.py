import bayesnet


def ask(var, value, evidence, bn):

    value_distribution = {}
    for values in [True, False]:
        evidences = evidence.copy()
        evidences[var] = values
        value_distribution[values] = enumerate_all(bn.variables, evidences)

    numerator = value_distribution[value]
    denominator = sum(value_distribution.values())

    fraction = numerator/denominator

    return fraction


def enumerate_all(variables, evidence):

    if len(variables) != 0:
        Y = variables[0]
        Y_left_values = variables[1:]

        for values in [True, False]:
            if Y.name in evidence and values == evidence[Y.name]:

                total = Y.probability(values, evidence) * \
                    enumerate_all(variables[1:], evidence)
                return total

        else:
            summation = 0
            for values in [True, False]:
                evidences = evidence.copy()
                evidences[Y.name] = values

                summation += Y.probability(values, evidences) * \
                    enumerate_all(Y_left_values, evidences)

            return summation

    return 1
