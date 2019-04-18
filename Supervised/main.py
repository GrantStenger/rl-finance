# Import Dependencies

# Constants
NUM_LINES = 10

def extract_prices():
    r_file = open('../data/AAPL.csv', 'r')
    w_file = open('../data/prices.csv', 'w')

    line = r_file.readline()
    while line:
        w_file.write(line.split(',')[2] + '\n')
        line = r_file.readline()

    r_file.close()
    w_file.close()

def convert_prices():
    r_file = open('../data/prices.csv', 'r')
    w_file = open('../data/convert_prices', 'w')

    title = r_file.readline()
    prev_line = r_file.readline()
    curr_line = r_file.readline()
    ups = 0
    ties = 0
    downs = 0
    while curr_line:
        if float(curr_line) > float(prev_line) + .01:
            w_file.write('1\n')
            ups += 1
        elif float(curr_line) < float(prev_line) - .01:
            w_file.write('-1\n')
            downs += 1
        else:
            w_file.write('0\n')
            ties += 1
        prev_line = curr_line
        curr_line = r_file.readline()

    print("Ups:", ups)
    print("Ties:", ties)
    print("Downs:", downs)

    r_file.close()
    w_file.close()

def compute_markov():
    r_file = open('../data/convert_prices', 'r')

    downs = 0
    ties = 0
    ups = 0

    down_down = 0
    down_tie = 0
    down_up = 0

    tie_down = 0
    tie_tie = 0
    tie_up = 0

    up_down = 0
    up_tie = 0
    up_up = 0

    total = 0

    prev_line = r_file.readline()
    curr_line = r_file.readline()
    while curr_line:
        total += 1
        if prev_line == '-1\n':
            downs += 1
            if curr_line == '-1\n':
                down_down += 1
            elif curr_line == '0\n':
                down_tie += 1
            else:
                down_up += 1
        elif prev_line == '0\n':
            ties += 1
            if curr_line == '-1\n':
                tie_down += 1
            elif curr_line == '0\n':
                tie_tie += 1
            else:
                tie_up += 1
        else:
            ups += 1
            if curr_line == '-1\n':
                up_down += 1
            elif curr_line == '0\n':
                up_tie += 1
            else:
                up_up += 1

        prev_line = curr_line
        curr_line = r_file.readline()

    r_file.close()

    print("Ups:", str(round(ups/total, 3)))
    print("Ties:", str(round(ties/total, 3)))
    print("Downs:", str(round(downs/total, 3)))
    print()
    print('down_down', str(round(down_down/downs, 3)))
    print('down_tie', str(round(down_tie/downs, 3)))
    print('down_up', str(round(down_up/downs, 3)))
    print()
    print('tie_down', str(round(tie_down/ties, 3)))
    print('tie_tie', str(round(tie_tie/ties, 3)))
    print('tie_up', str(round(tie_up/ties, 3)))
    print()
    print('up_down', str(round(up_down/ups, 3)))
    print('up_tie', str(round(up_tie/ups, 3)))
    print('up_up', str(round(up_up/ups, 3)))
    print()
    print('total', total)

def pre_processing():

    WINDOW_SIZE = 2
    NUM_ITERATIONS = 10
    TRAIN_TEST_RATIO = 0.8

    x = []
    y = []

    window = []

    r_file = open('../data/convert_prices', 'r')
    for i in range(NUM_ITERATIONS):
        window = []
        for j in range(WINDOW_SIZE):
            line = r_file.readline()
            window.append(line[:-1])
        x.append(window)
        next = r_file.readline()[:-1]
        y.append(next)

    r_file.close()

    training_size = int(TRAIN_TEST_RATIO * len(x))

    x_train = x[:training_size]
    x_test = x[training_size:]
    y_train = y[:training_size]
    y_test = y[training_size:]

    return x_train, y_train, x_test, y_test

def neural_network():

    x_train, y_train, x_test, y_test = pre_processing()

    for i in range(len(x_train)):
        print(x_train[i], y_train[i])
    print()
    for i in range(len(x_test)):
        print(x_test[i], y_test[i])

def main():
    # extract_prices()
    # convert_prices()
    # compute_markov()
    neural_network()

if __name__ == "__main__":
    main()
