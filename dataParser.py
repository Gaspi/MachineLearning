# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 15:58:38 2014

@author: Sylvain
"""

import numpy as np
from datetime import date
import time
import csv
import operator
from random import random
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

data_conversion = np.genfromtxt("Forex_2000_2014_FINAL enhanced_for_PYTHON_2.txt",delimiter=",")
scores = ["3-0","3-1","3-2","0-3","1-3","2-3","2-0","2-1","0-2","1-2","RW","RL"]




"""  Data parser functions  """

# Reads the csv and creates, for each player, a list containing all his games.
def parseAndGetData():
    data = dict()
    intFields = ['IDPlayer','IDTournament','IDOpponent','Indoor','SecondServeReturnPointsWon','Retirement','BreakPointsConvertedTotal','SecondServePointsWon','BreakPointsConverted','y','TotalReturnPointsWon','FirstServeReturnPointsWonTotal','Draw','TotalPointsWon','RoundNumber','Timestamp','SecondServePointsWonTotal','FirstServePointsWonTotal','TotalReturnPointsWonTotal','BreakPointsSavedTotal','ServiceGamesPlayed','SecondServeReturnPointsWonTotal','t','TotalServicePointsWonTotal','Aces','TotalServicePointsWon','DoubleFaults','FirstServeReturnPointsWon','ReturnGamesPlayed','FirstServe','FirstServeTotal','Win','r','FirstServePointsWon','TotalPointsWonTotal','Year','Duration','BreakPointsSaved']
    numberPlayers = 0
    numberTournaments = 0
    with open("matches.csv","r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter='|')
        for game in reader:
            for key in intFields:
                game[key] = int(game[key])
                if game[key] == -1:
                    game[key] = np.nan
            if not game["IDPlayer"] in data:
                data[game["IDPlayer"]] = []
            data[game["IDPlayer"]].append(game)    
    for player in data:
        data[player].sort(key=operator.itemgetter('Timestamp'))
    players = []
    with open("players.csv","r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter='|')
        for player in reader:
            numberPlayers += 1
            for key in ['ID','DayBirth','MonthBirth','YearBirth','Height','Weight','RightHanded','TurnedPro']:
                player[key] = int(player[key])
                if player[key] == -1:
                    player[key] = np.nan
            players.append(player)
    players.sort(key=operator.itemgetter('ID'))
    with open("tournaments.csv","r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter='|')
        for tournament in reader:
            numberTournaments += 1
    return data, players, numberTournaments, numberPlayers
    


"""  General functions  """
                        
# Allows to generate a binary vector representing a discrete value.
def transformDiscreteValueToVector(value, dim):
    vector = [0]*dim
    vector[value] = 1
    return vector
    

# Converts a prize and a date into a dollar price with the conversion rate at the time of the game. 
def computePrizeInDollars(prize, date):
    prize = unicode(prize)
    currency = prize[:1]
    amount_dollar = prize[1:]
    amount_euro = prize[3:]
    d = date.split('.')
    if currency == "$":
        return int(amount_dollar)
    elif currency == 'Ç':
        return int(amount_dollar)*forex(int(d[2]),int(d[1]),int(d[0]))
    else:
        return int(amount_euro)*forex(int(d[2]),int(d[1]),int(d[0]))
        

# Returns the conversion rate €-$ in a given date.        
def forex(Y,M,D):
    ref=str(Y)+str(date(Y, M, D).isocalendar()[1])
    for i in range(data_conversion.shape[0]):
            if int(data_conversion[i,0])==int(ref):
                return data_conversion[i,1] 
                
                
# Returns the vector representing the surface.
def returnSurfaceVector(surface):
    vector = [0]*4
    if surface=='Clay':
        vector[0]=1
    elif surface=='Grass':
        vector[1]=1
    elif surface == 'Carpet':
        vector[2]=1
    else:
        vector[3]=1
    return vector
    

# Print a string containing the successive names of the labels.
def printLabelsVector():
    return "\t".join(['Clay','Grass','Carpet','Other','Timestamp','Draw','TournamentPrize','RoundNumber','Month','Year'])
    
    
# Computes the age of a player given his birthday.
def computeAge(day, month, year):
    if day == np.nan or month == np.nan or year == np.nan:
        return np.nan
    today = (time.strftime('%d.%m.%Y',time.localtime())).split('.')
    if (month>int(today[1]) or (month==int(today[1]) and day>int(today[0]))):
        return int(today[2])-year-1
    else:
        return int(today[2])-year
        
    
    


"""  Match vectors generation functions  """

# From a given game, creates the binary vector representing the score.
def createScoreVector(game):
    vector = [0]*12
    index = computeScoreLabel(game)
    vector[index] = 1
    return vector


# Computes the label corresponding to the score of the game (ie the index of the 1 in the vector representing the score).
def computeScoreLabel(game):
    winnerScores = game['WinnerScores'].replace('[','').replace(']','').replace('\'','').split(',')
    loserScores = game['LoserScores'].replace('[','').replace(']','').replace('\'','').split(',')
    retirement = game['Retirement']
    win = game['Win']
    if retirement:
        if win:
            return 10
        else:
            return 11
    setsWinner = 0
    setsLoser = 0    
    for i in range(len(winnerScores)):
        ws = int(winnerScores[i])
        ls = int(loserScores[i])
        if (ws > ls):
            setsWinner += 1
        elif (ws < ls):
            setsLoser += 1
        else:
            pass
    if win:
        if setsWinner == 3:
            return setsLoser
        elif setsWinner == 2:
            return 6+setsLoser
        else:
            pass
    else:
        if setsWinner == 3:
            return 3+setsLoser
        elif setsWinner == 2:
            return 8+setsLoser
        else:
            return -1


# Generates a vector containing the data about the tournament.    
def generateGameInformationsVector(game, numberTournaments):
    date = game['TournamentStart'].split('.')
    tournament = transformDiscreteValueToVector(game['IDTournament'], numberTournaments) # To implement
    surface = returnSurfaceVector(game['Surface']) # To implement
    data = [int(game['Timestamp']/10), game['Draw'], game['Indoor'],computePrizeInDollars(game['TournamentPrize'], game['TournamentStart']), game['RoundNumber'], int(date[1]), int(date[2])]
    return tournament+surface+data
    

# Generates a vector containing the data about a player.    
def generatePlayerInformation(playerID, numberPlayers, players, surface, timestamp, data):
    player=transformDiscreteValueToVector(playerID, numberPlayers)
    dataPlayer = players[playerID]
    globalData = [computeAge(dataPlayer['DayBirth'], dataPlayer['MonthBirth'], dataPlayer['YearBirth']), dataPlayer['Height'], dataPlayer['Weight'], dataPlayer['RightHanded'], dataPlayer['TurnedPro']]
    lastMatches = generateLastMatchesData(playerID, 5, timestamp, data)
    lastSurfaceMatches = generateLastMatchesData(playerID, 5, timestamp, data, surface=surface)
    return player+globalData+lastMatches+lastSurfaceMatches
    

# Returns a vector containing the data about the last n matches of a player before a given date (represented by a timestamp), eventually played in a given surface.    
def generateLastMatchesData(playerID, n, timestamp, data, surface=None):
    vector = []    
    if surface==None:
        playerData = data[playerID]
    else:
        playerData = [i for i in data[playerID] if i['Surface']==surface]
    for i in range(len(playerData)):
        if playerData[i]['Timestamp'] >= timestamp:
            i -= 1
            break
    i += 1
    firstMatch = i-n    
    for i in range(n):
        if (firstMatch+i)<0:
            vector += generateMatchData(None)
        else:
            vector += generateMatchData(playerData[firstMatch+i])
    return vector
    

# Generates a vector containing the data about a match.
def generateMatchData(game):    
    if game == None:
        return [np.nan]*27
    else:
        vector = [0]*27
        vector[0]=game['Duration']
        vector[1]=game['Win']
        vector[2]=game['RoundNumber']
        vector[3]=game['Aces']
        vector[4]=game['BreakPointsConverted']
        vector[5]=game['BreakPointsConvertedTotal']
        vector[6]=game['BreakPointsSaved']
        vector[7]=game['BreakPointsSavedTotal']
        vector[8]=game['DoubleFaults']
        vector[9]=game['FirstServe']
        vector[10]=game['FirstServePointsWon']
        vector[11]=game['FirstServePointsWonTotal']
        vector[12]=game['FirstServeReturnPointsWon']
        vector[13]=game['FirstServeReturnPointsWonTotal']
        vector[14]=game['FirstServeTotal']
        vector[15]=game['ReturnGamesPlayed']
        vector[16]=game['SecondServePointsWon']
        vector[17]=game['SecondServePointsWonTotal']
        vector[18]=game['SecondServeReturnPointsWon']
        vector[19]=game['SecondServeReturnPointsWonTotal']
        vector[20]=game['ServiceGamesPlayed']
        vector[21]=game['TotalPointsWon']
        vector[22]=game['TotalPointsWonTotal']
        vector[23]=game['TotalReturnPointsWon']
        vector[24]=game['TotalReturnPointsWonTotal']
        vector[25]=game['TotalServicePointsWon']
        vector[26]=game['TotalServicePointsWonTotal']
        return vector
        

# Generates the global vector representing a match.
def generateMatchVector(game, numberTournaments, numberPlayers, data, players):
    return generateGameInformationsVector(game, numberTournaments)+generatePlayerInformation(game['IDPlayer'], numberPlayers, players, game['Surface'], game['Timestamp'], data)+generatePlayerInformation(game['IDOpponent'], numberPlayers, players, game['Surface'], game['Timestamp'], data)


# From a given database, returns the matrix of all the data.
def generateAllMatchesVector(numberTournaments, numberPlayers, data, players):
    matrix = []
    labels = []
    for player in data:
        for game in data[player]:
            matrix.append(generateMatchVector(game, numberTournaments, numberPlayers, data, players))
            labels.append(createScoreVector(game))
    return matrix, labels


# For testing, generates a learning dataset and a validation dataset.
def generateTestingData(numberTournaments, numberPlayers, data, players):
    learningData = []
    validationData = []
    learningLabel = []
    validationLabel = []
    for player in data:
        for game in data[player]:
            if random() < 0.5:
                learningData.append(generateMatchVector(game, numberTournaments, numberPlayers, data, players))
                learningLabel.append(computeScoreLabel(game))
            else:
                validationData.append(generateMatchVector(game, numberTournaments, numberPlayers, data, players))
                validationLabel.append(computeScoreLabel(game))
    return learningData, learningLabel, validationData, validationLabel



""" Preprocessing functions """

# Completes the data when there are unknown values.
def completeData(incompleteData):
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    return imputer.fit_transform(incompleteData)
    
    
# Normalizes the values in the vectors.
def normalizeData(unnormalizedData):
    return scale(unnormalizedData)
    
    
# Applies a dimension reduction to the data.
def reduceDimensions(originalData, n_dimensions):
    pca = PCA(n_components=n_dimensions)
    return pca.fit_transform(originalData)
                   



if __name__ == '__main__':
    data, players, numberTournaments, numberPlayers = parseAndGetData()
    learningData, learningLabel, validationData, validationLabel = generateTestingData(numberTournaments, numberPlayers, data, players)
    
    tr = reduceDimensions(normalizeData(completeData(learningData)),500)
    with open("result","w") as file_:
        for line in tr:
            file_.write("\t".join([str(i) for i in line])+"\n")

        
