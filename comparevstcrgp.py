for file in file_dash_human:
    filename = file.split(".")[0]
    data = pd.read_csv(path_dash_human + file, encoding='gbk')
    X = feature(data)
    y = classes(data)
    X = np.array(X)
    empty_TT = []
    empty_GP = []
    sfolder = StratifiedShuffleSplit(n_splits=5, random_state=111, test_size=test_size)
    for train_idx, test_idx in sfolder.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if np.sum(y_train == 0) == np.sum(y_train == 1):
            print(filename)
            print(test_size)
            print(Counter(y_train))
            print(Counter(y_test))
            TT_list = []
            for i in range(5):
                TT.fit(X_train, y_train, X_test)
                acc_U = TT.score(X_test, y_test)
                TT_list.append(acc_U)
            TT_mean = np.mean(TT_list)
            empty_TT.append(TT_mean)
            acc_gp = tcrgp(data, train_idx, test_idx, filename)
            empty_GP.append(acc_gp)
            print(train_idx[:10])
            print("TT:", TT_mean)
            print("TCRGP", acc_gp)
            print("========================")
    print(filename)
    print(test_size)
    print(np.mean(empty_TT))
    print(np.mean(empty_GP))
    print("========================")
