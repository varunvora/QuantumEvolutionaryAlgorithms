from numpy import random, concatenate, array


def sda(num, clusterMembersNum=100):
    "This function will generate random datasets : sda1,sda2,sda3"
    seed = 0
    random.seed(seed)
    if num == 1:
        "generating sda1 according to its table in essay"
        dataset = concatenate([random.uniform(0, 20, (clusterMembersNum, 2)),
                               random.uniform(40, 60, (clusterMembersNum, 2)),
                               random.uniform(80, 100, (clusterMembersNum, 2))])
    elif num == 2:
        "generating sda2 according to its table in essay"
        dataset = concatenate([random.uniform(0, 20, (clusterMembersNum, 2)),
                               random.uniform(40, 60, (clusterMembersNum, 2)),
                               random.uniform(80, 100, (clusterMembersNum, 2)),
                               array([[random.uniform(0, 20), random.uniform(80, 100)] for i in
                                      range(clusterMembersNum)])])
    else:
        "generating sda3 according to its table in essay"
        dataset = concatenate([random.uniform(0, 20, (clusterMembersNum, 2)),
                               random.uniform(40, 60, (clusterMembersNum, 2)),
                               random.uniform(80, 100, (clusterMembersNum, 2)),
                               array([[random.uniform(80, 100), random.uniform(0, 20)] for i in
                                      range(clusterMembersNum)]),
                               array([[random.uniform(0, 20), random.uniform(180, 200)] for i in
                                      range(clusterMembersNum)]),
                               array([[random.uniform(180, 200), random.uniform(0, 20)] for i in
                                      range(clusterMembersNum)]),
                               array([[random.uniform(180, 200), random.uniform(80, 100)] for i in
                                      range(clusterMembersNum)]),
                               array([[random.uniform(180, 200), random.uniform(180, 200)] for i in
                                      range(clusterMembersNum)])])
    return array(dataset)


if __name__ == "__main__":
    a = sda(1)
    print(a)