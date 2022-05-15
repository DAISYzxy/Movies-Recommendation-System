import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse import vstack
import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings

warnings.filterwarnings('ignore')

np.random.seed(5001)


class TotalDataset(Dataset):

    def __init__(self, ratingData):
        self.csv = ratingData
        self.user_ids = list(self.csv.UserID - 1)
        self.movie_ids = list(self.csv.MovieID - 1)
        self.ratings = list(self.csv.Rating)
        self.user_nums = np.max(self.user_ids) + 1
        self.movie_nums = np.max(self.movie_ids) + 1

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        return {'user': self.user_ids[index],
                'movie': self.movie_ids[index],
                'rating': self.ratings[index]
                }

    def get_user_number(self):
        return self.user_nums

    def get_movie_number(self):
        return self.movie_nums


class GCNLayer(Module):

    def __init__(self, inD, outD):
        super(GCNLayer, self).__init__()
        self.inD = inD
        self.outD = outD
        self.linear1 = torch.nn.Linear(in_features=inD, out_features=outD)
        self.linear2 = torch.nn.Linear(in_features=inD, out_features=outD)

    def forward(self, lapMat, loopMat, features):
        L1 = lapMat + loopMat
        L2 = lapMat
        inter_features = torch.mul(features, features)
        inter1 = self.linear1(torch.sparse.mm(L1, features))
        inter2 = self.linear2(torch.sparse.mm(L2, inter_features))
        return inter1 + inter2


class NGCF(Module):

    def __init__(self, userNum, itemNum, ratingData, embedSize=64, layers=[64, 64]):
        super(NGCF, self).__init__()
        self.userNum = userNum
        self.itemNum = itemNum

        self.userEmbed = nn.Embedding(userNum, embedSize)
        self.itemEmbed = nn.Embedding(itemNum, embedSize)

        self.GCNlayers = torch.nn.ModuleList()

        self.lapMat = self.getLapMat(ratingData)
        self.loopMat = self.getLoopMat(self.userNum + self.itemNum)

        self.linear1 = nn.Linear(in_features=layers[-1] * (len(layers)) * 2, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=32)
        self.linear3 = nn.Linear(in_features=32, out_features=1)

        for inD, outD in zip(layers[:-1], layers[1:]):
            self.GCNlayers.append(GCNLayer(inD, outD))

    def getLapMat(self, ratingData):
        rt_item = ratingData['MovieID'] + self.userNum

        Mat1 = coo_matrix((ratingData['Rating'], (ratingData['UserID'], ratingData['MovieID'])))
        Mat1 = Mat1.transpose()
        Mat1.resize((self.itemNum, self.userNum + self.itemNum))

        Mat2 = coo_matrix((ratingData['Rating'], (ratingData['UserID'], rt_item)))

        A = sparse.vstack([Mat2, Mat1])

        sumArr = (A > 0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag, -0.5)
        D = sparse.diags(diag)

        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        index = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        sparseL = torch.sparse.FloatTensor(index, data)

        return sparseL

    def getLoopMat(self, num):
        index = torch.LongTensor([[k for k in range(0, num)], [j for j in range(0, num)]])
        data = torch.FloatTensor([1] * num)
        return torch.sparse.FloatTensor(index, data)

    def getFeatureMat(self):
        indexUser = torch.LongTensor([i for i in range(self.userNum)])
        indexItem = torch.LongTensor([i for i in range(self.itemNum)])

        userEmbed = self.userEmbed(indexUser)
        itemEmbed = self.itemEmbed(indexItem)
        features = torch.cat([userEmbed, itemEmbed], dim=0)

        return features

    def forward(self, userID, itemID):
        itemID = itemID + self.userNum
        userID = list(userID.cpu().data)
        itemID = list(itemID.cpu().data)

        features = self.getFeatureMat()
        final = features.clone()

        for gcn in self.GCNlayers:
            features = gcn(self.lapMat, self.loopMat, features)
            features = nn.LeakyReLU()(features)
            final = torch.cat([final, features.clone()], dim=1)

        userEmbed = final[userID]
        itemEmbed = final[itemID]

        embed = torch.cat([userEmbed, itemEmbed], dim=1)
        embed = nn.ReLU()(self.linear1(embed))
        embed = nn.ReLU()(self.linear2(embed))
        embed = self.linear3(embed)
        prediction = embed.flatten()

        return prediction


def train(data, model, opti, loss, device):
    model.train()
    num = 0.

    for i, batch in enumerate(data):
        opti.zero_grad()
        prediction = model(batch['user'].to(device), batch['movie'].to(device))
        loss_batch = loss(batch['rating'].float().to(device), prediction)
        loss_batch.backward()
        opti.step()
        num += loss_batch.item()

    return num / len(data)


def evaluate(data, model, loss, device):
    model.eval()
    num = 0.
    results = []

    with torch.no_grad():
        for i, batch in enumerate(data):
            prediction = model(batch['user'].to(device), batch['movie'].to(device))
            loss_batch = loss(batch['rating'].float().to(device), prediction)
            num += loss_batch.item()
            results += prediction

    return num / len(data), results


def trainer(trainData, valData, model, opti, loss, epoch, model_name, verbose, device):
    train_loss = []
    val_loss = []
    best_results = np.inf
    best_recom_results = []

    for i in range(epoch):
        start = time.time()
        train_results = train(trainData, model, opti, loss, device)
        val_results, recom_results = evaluate(valData, model, loss, device)

        if (verbose):
            print("Epoch {} | Train Loss: {:.3f} - Val Loss: {:.3f} - in {:.3f} mins.".format(i + 1, train_results,
                                                                                              val_results, (
                                                                                                      time.time() - start) / 60))

        if val_results < best_results:
            best_results = val_results
            best_recom_results = recom_results

        train_loss.append(train_results)
        val_loss.append(val_results)

    return best_recom_results


# def Recommendation_NGCF(ratingArr, movieids, num_movie, df):
def Recommendation_NGCF(ratingArr, movieids, num_movie, root_dir):
    # ratingDataRead=df
    ratingDataRead = pd.read_csv(root_dir + 'MovieRatingUser.csv')

    recomNum = len(movieids)
    total_movie = ratingDataRead['MovieID'].unique()
    test_movie = list(set(total_movie) - set(movieids))

    trainNum = len(list(ratingDataRead.Rating)) + recomNum
    testNum = len(test_movie)

    ratingData = {'MovieID': list(ratingDataRead.MovieID) + list(movieids) + test_movie,
                  'UserID': list(ratingDataRead.UserID) + [max(ratingDataRead.UserID) + 1 for i in
                                                           range(len(total_movie))],
                  'Rating': list(ratingDataRead.Rating) + ratingArr + [0 for i in range(len(test_movie))]}

    ratingData = pd.DataFrame(ratingData)

    dataset = TotalDataset(ratingData)

    ratingData['UserID'] = ratingData['UserID'] - 1
    ratingData['MovieID'] = ratingData['MovieID'] - 1

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [trainNum, testNum])

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    bs = 512
    layer = [64, 64]
    embedding_dim = 64
    epoch = 1
    lr = 1e-3

    trainData = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
    valData = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, drop_last=True)

    # trainData = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=1, drop_last=True)
    # valData = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=1, drop_last=True)

    model = NGCF(dataset.get_user_number(), dataset.get_movie_number(), ratingData, embedding_dim, layers=layer).to(
        device)
    opti = Adam(model.parameters(), lr=lr)
    loss = nn.L1Loss()

    out_results = trainer(trainData, valData, model, opti, loss, epoch, 'NGCF', False, device)

    max_out = np.max(out_results)
    min_out = np.min(out_results)
    out_results = [(x - min_out) / (max_out - min_out) * 5.0 for x in out_results]

    tmp = dict(zip(out_results, test_movie))
    tmp = sorted(tmp.items(), reverse=True)

    return [x[1] for x in tmp][0:num_movie], [float(x[0]) for x in tmp][0:num_movie]

# root_dir = '/Users/asteriachiang/Documents/5001_Foundations_of_Data_Analytics/model/'
# l1,l2=Recommendation_NGCF([1.0,2.0,3.0,4.0,5.0,1.0,2.0,3.0,4.0,5.0],[296,1225,1288,1027,1343,10,380,1320,1030,356],10,root_dir)
# print(l1)
# print(l2)
