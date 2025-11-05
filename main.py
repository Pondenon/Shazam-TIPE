from audioExtract import AudioExtract
from songs import Song
from auxiliaries import database, database_, sampleRate, numberFFTPoints, sampleRates, fFFT, data_to_add, songData, min, max, convertTabToFloat, redTer, whiTer, greTer, oraTer, cyaTer, purBac, blaBac, maxList, absolute, featuresToString, compareElementsPerFootprint

import numpy as np
import time as T

global debugging
debugging = False

global numberFreq
numberFreq = fFFT[numberFFTPoints] #short : 2048 / medium : 4096

global freq 
freq = sampleRates[sampleRate] #44 : 44100 / 22 : 22050



def joinSongs(songs : list[Song], footprints : list[Song], footprint=0):
    """
    Entrée: un tableau de chansons ayant certaines signatures, un tableau avec d'autres empreintes et l'empreinte du deuxième tableau\n
    Sortie: ajoute à chaque chanson du premier tableau l'empreinte du deuxième tableau
    """
    for i in range(len(songs)):
        if footprint == 0:
            songs[i].audioExtract.matrix = footprints[i].audioExtract.matrix
        elif footprint == 1:
            songs[i].audioExtract.fingerprint = footprints[i].audioExtract.fingerprint
            songs[i].audioExtract.fingerprintMatrix = footprints[i].audioExtract.fingerprintMatrix
        else: #footprint == 2
            songs[i].audioExtract.signature = footprints[i].audioExtract.signature
            songs[i].audioExtract.signatureMatrix = footprints[i].audioExtract.signatureMatrix
    return songs

def getExtractInfo(fileName):
    """
    Entrée: le nom d'un extrait audio\n
    Sortie: renvoie un extrait audio contenant toutes ses empreintes
    """
    extract = AudioExtract(fileName, "Extracts/")
    extract.load()
    extract.createFingerprint()
    extract.generateSignature()
    return extract

def sortFootprints(d, ind, footprint=0):
    """
    Entrée: un tableau d'empreintes et un tableau d'elements\n
    Sortie: renvoie le tableau d'elements triés selon le tableau d'empreintes triés
    """
    dist = d.copy()
    index = ind.copy()
    for i in range(len(dist) - 1):
        for j in range(i + 1, len(dist)):
            if compareElementsPerFootprint(dist[i], dist[j], footprint) == 1:
                index[i], index[j] = index[j], index[i]
                dist[i], dist[j] = dist[j], dist[i]
    return dist, index

def calculateScore(extract : AudioExtract, songs : list[Song], comp, footprint=0):
    """
    Entrée: un extrait audio, un tableau de chansons et un tableau d'empreintes dans l'ordre décroissant de l'empreinte\n
    Sortie: renvoie un tableau dans l'ordre décroissant contenant le score de chaque chanson
    """
    score = []
    maxMatches = max(comp[0], 0.00001)
    length = extract.matrix.nb_col
    percent = 0
    if footprint == 1:
        signatureLength = len(extract.fingerprint)
    elif footprint == 2:
        signatureLength = len(extract.signature)
    for i in range(len(comp)):
        matches = comp[i]
        songLength = songs[i].audioExtract.matrix.nb_col
        songScore = (comp[i] / signatureLength) * (length / songLength) * (matches / maxMatches) * 100 * matches
        score.append(songScore)
        percent += songScore
    coeff = 100 / percent if percent != 0 else 0
    return score, coeff


def loadSongMatrixFromText(fileName):
    """
    Entrée: le nom d'un fichier\n
    Sortie: renvoie la matrice de la chanson correspondante
    """
    f = open(f"/Users/jeremyvenencie/Documents/CPGE/TIPE/Code/Data_Base/{numberFreq}/{freq}kHz/Text_Matrice/{fileName}.txt", "r")
    title = f.readline()[:-1]
    artist = f.readline()[:-1]
    feats = f.readline().split("|")
    genre = f.readline()[:-1]
    if feats == ['']:
        feats = []
    else:
        feats[-1] = feats[-1][:-1]
    song = Song(title, artist, feats, genre)
    filename = f.readline()[:-1]
    size = f.readline().split("x")
    size[-1] = size[-1][:-1]
    f.readline()
    matrix = []
    for _ in range(int(size[0])):
        t = f.readline().split("|")
        t[-1] = t[-1][:-1]
        matrix.append(convertTabToFloat(t))
    song.loadMatrix(matrix)
    song.fileName = filename
    song.audioExtract.fingerprintMatrix.load(np.zeros((int(size[0]), int(size[1]))))
    song.audioExtract.signatureMatrix.load(np.zeros((int(size[0]), int(size[1]))))
    f.close()
    return song

def getSongsMatrix(tab):
    """
    Entrée: un tableau de noms de fichiers\n
    Sortie: renvoie un tableau contenant les matrices des chansons
    """
    stLM = T.time()
    t = []
    print("\nLoading songs' matrix values ...")
    for i in range(len(tab)):
        t.append(loadSongMatrixFromText(tab[i]))
    print("All the songs' values are loaded.")
    etLM = T.time()
    print(f"\tExecuted in " + purBac + f"{(etLM - stLM):.2f}" + blaBac + " seconds\n")
    return t

def getMatrixDistances(extract : AudioExtract, tab : list[Song], d : int):
    """
    Entrée: un extrait audio, un tableau de chansons et une distance\n
    Sortie: renvoie un tableau contenant les distances (norme d) entre l'extrait audio et les chansons
    """
    stCM = T.time()
    t = []
    print("\nCalculating distances ...")
    for i in range(len(tab)):
        dist = extract.compare(tab[i].audioExtract, d)
        t.append(dist)
    print("All distances are calculated")
    etCM = T.time()
    print(f"\tExecuted in " + purBac + f"{(etCM - stCM):.2f}" + blaBac + " seconds\n")
    return t

def mostProbableSongMethod1(extract : AudioExtract, songs : list[Song], d):
    """
    Entrée: le nom d'un extrait audio, un tableau de chansons et une distance\n
    Sortie: renvoie le tableau de chansons triés dans l'ordre décroissant de chance selon la matrice d'être la chanson de l'extrait
    """
    distances = getMatrixDistances(extract, songs, d)
    distances, songs = sortFootprints(distances, songs)
    feats = featuresToString(songs[0].feats)
    print(f"\nThe distance used is '{d}'\n")
    print(f"\nThe extract is most likely to be from the song: \n\t - " + cyaTer + f"{songs[0].title}" + whiTer + f"\n\t - by " + cyaTer + f"{songs[0].artist}" + whiTer + (f"\n\t - {feats}" if feats != '' else '') + f"\n\t - " + oraTer + f"{songs[0].genre}" + whiTer + f"\n\t - distance = " + greTer + f"{distances[0]}" + whiTer + f"\n")
    print("\n")
    for i in range(len(songs)):
        print(f"{i+1} - " + cyaTer + f"{songs[i].title} " + whiTer + f"by " + cyaTer + f"{songs[i].artist} " + whiTer + featuresToString(songs[i].feats) + "(" + oraTer + f"{songs[i].genre}" + whiTer + ") " + "with " + greTer + f"{distances[i]} " + whiTer + f"distance.")
    print("\n")
    if debugging:
        copy = songs[0].audioExtract.adapt(extract)
        extract.displayComparison(copy)
        extract.displayComparison(songs[-1].audioExtract)
    return songs


def loadSongFingerprintFromText(fileName):
    """
    Entrée: le nom d'un fichier\n
    Sortie: renvoie l'empreinte de la chanson correspondante
    """
    f = open(f"/Users/jeremyvenencie/Documents/CPGE/TIPE/Code/Data_Base/{numberFreq}/{freq}kHz/Text_Empreinte/{fileName}.txt", "r")
    title = f.readline()[:-1]
    artist = f.readline()[:-1]
    feats = f.readline().split("|")
    genre = f.readline()[:-1]
    if feats == ['']:
        feats = []
    else:
        feats[-1] = feats[-1][:-1]
    song = Song(title, artist, feats, genre)
    filename = f.readline()[:-1]
    size = f.readline().split("x")
    size[-1] = size[-1][:-1]
    length = int(f.readline())
    f.readline() #blank space in txt file
    signature = []
    for i in range(length):
        ft = f.readline().split("|")
        tupl = (int(ft[0]), int(ft[1]), float(ft[2]))
        signature.append(tupl)
    song.loadFingerprint(signature, int(size[0]), int(size[1]))
    song.fileName = filename
    song.audioExtract.matrix.load(np.zeros((int(size[0]), int(size[1]))))
    song.audioExtract.signatureMatrix.load(np.zeros((int(size[0]), int(size[1]))))
    f.close()
    return song

def getSongsFingerprint(tab):
    """
    Entrée: un tableau de noms de fichiers\n
    Sortie: renvoie un tableau contenant les empreintes des chansons
    """
    stLF = T.time()
    t = []
    print("\nLoading songs' signatures ...")
    for i in range(len(tab)):
        t.append(loadSongFingerprintFromText(tab[i]))
    print("All the songs' signatures are loaded.")
    etLF = T.time()
    print(f"\tExecuted in " + purBac + f"{(etLF - stLF):.2f}" + blaBac + " seconds\n")
    return t

def getFingerprintMatches(extract : AudioExtract, tab : list[Song]):
    """
    Entrée: un extrait audio et un tableau de chansons\n
    Sortie: renvoie un tableau contenant la comparaison des empreintes des chansons
    """
    stCF = T.time()
    compMax = []
    deltaMax = []
    print("\nCalculating signatures ...")
    for i in range(len(tab)):
        matches, delta = extract.fingerprintComparingAlignement(tab[i].audioExtract)
        compMax.append(matches) #methode faites avant
        deltaMax.append(delta)
    print("All signatures are calculated")
    etCF = T.time()
    print(f"\tExecuted in " + purBac + f"{(etCF - stCF):.2f}" + blaBac + " seconds\n")
    return compMax, deltaMax

def mostProbableSongMethod2(extract : AudioExtract, songs : list[Song]):
    """
    Entrée: le nom d'un extrait audio et un tableau de chansons\n
    Sortie: renvoie le tableau de chansons triés dans l'ordre décroissant de chance selon l'empreinte d'être la chanson de l'extrait
    """
    compMax, deltaMax = getFingerprintMatches(extract, songs)
    comp, deltaMax = sortFootprints(compMax, deltaMax, 1)
    compMax, songs = sortFootprints(compMax, songs, 1)
    score, certainty = calculateScore(extract, songs, compMax, 1)
    feats = featuresToString(songs[0].feats)
    #if (2 * score[0]) >= score[1]:
    print(f"\nThe footprint used is the fingerprint: coordinates of local maximum peaks.")
    print(f"\nThe extract is most likely to be from the song (certainty = " + greTer + f"{round(score[0] * certainty, 2)}" + whiTer + f"):\n\t - " + cyaTer + f"{songs[0].title}" + whiTer + f"\n\t - by " + cyaTer + f"{songs[0].artist}" + whiTer + (f"\n\t - {feats}" if feats != '' else '') + f"\n\t - " + oraTer + f"{songs[0].genre}" + whiTer + f"\n\t - " + greTer + f"{compMax[0]} " + whiTer + f"matched peaks with " + greTer + f"{score[0]}" + whiTer + " score.\n")
    for i in range(len(songs)):
        print(f"{i+1} - " + cyaTer + f"{songs[i].title} " + whiTer + f"by " + cyaTer + f"{songs[i].artist} " + whiTer + featuresToString(songs[i].feats) + "(" + oraTer + f"{songs[i].genre}" + whiTer + ") " + "with " + greTer + f"{compMax[i]} " + whiTer + f"peaks matched with a " + greTer + f"{round(score[i] * certainty, 2)} " + whiTer + f"certainty.")
    print()
    print(f"delta = +{round(deltaMax[0]/44, 2)}s")
    print()
    #else:
        #print(f"\nUnfortunately, the extract was not recognized with a sufficient level of certainty. (only {score[0] * certainty})\n")
    if debugging:
        copy = songs[0].audioExtract.adapt(extract)
        extract.displayComparison(copy)
        copyMax = songs[0].audioExtract.adapt(extract)
        extract.displayComparison(copyMax)
    return songs, deltaMax


def loadSongSignatureFromText(fileName):
    """
    Entrée: le nom d'un fichier\n
    Sortie: renvoie la signature de la chanson correspondante
    """
    f = open(f"/Users/jeremyvenencie/Documents/CPGE/TIPE/Code/Data_Base/{numberFreq}/{freq}kHz/Text_Signature/{fileName}.txt", "r")
    title = f.readline()[:-1]
    artist = f.readline()[:-1]
    feats = f.readline().split("|")
    genre = f.readline()[:-1]
    if feats == ['']:
        feats = []
    else:
        feats[-1] = feats[-1][:-1]
    song = Song(title, artist, feats, genre)
    filename = f.readline()[:-1]
    size = f.readline().split("x")
    size[-1] = size[-1][:-1]
    length = int(f.readline())
    f.readline()
    signature = []
    for _ in range(length):
        ft = f.readline().split("|")
        ft = (float(ft[0]), int(ft[1]), int(ft[2]))
        signature.append(ft)
    song.loadSignature(signature, int(size[0]), int(size[1]))
    song.fileName = filename
    song.audioExtract.matrix.load(np.zeros((int(size[0]), int(size[1]))))
    song.audioExtract.fingerprintMatrix.load(np.zeros((int(size[0]), int(size[1]))))
    f.close()
    return song

def getSongsSignature(tab):
    """
    Entrée: un tableau de noms de fichiers\n
    Sortie: renvoie un tableau contenant les signatures des chansons
    """
    stLS = T.time()
    t = []
    print("\nLoading songs' second signatures ...")
    for i in range(len(tab)):
        t.append(loadSongSignatureFromText(tab[i]))
    print("All the songs' second signatures are loaded.")
    etLS = T.time()
    print(f"\tExecuted in " + purBac + f"{(etLS - stLS):.2f}" + blaBac + " seconds\n")
    return t

def getSignatureMatches(extract : AudioExtract, tab : list[Song]):
    """
    Entrée: un extrait audio et un tableau de chansons\n
    Sortie: renvoie un tableau contenant la comparaison des signatures des chansons
    """
    stCS = T.time()
    compMax = []
    deltaMax = []
    print("\nCalculating signatures ...")
    for i in range(len(tab)):
        #matches = tab[i].audioExtract.compareSignatures(extract)
        matches, delta = extract.signatureComparingAlignement(tab[i].audioExtract)
        compMax.append(matches)
        deltaMax.append(delta)
    print("All signatures are calculated")
    etCS = T.time()
    print(f"\tExecuted in " + purBac + f"{(etCS - stCS):.2f}" + blaBac + " seconds\n")
    return compMax, deltaMax

def mostProbableSongMethod3(extract : AudioExtract, songs : list[Song]):
    """
    Entrée: le nom d'un extrait audio et un tableau de chansons\n
    Sortie: renvoie le tableau de chansons triés dans l'ordre décroissant de chance selon la signature d'être la chanson de l'extrait
    """
    compMax, deltaMax = getSignatureMatches(extract, songs)
    comp, deltaMax = sortFootprints(compMax, deltaMax, 2)
    compMax, songs = sortFootprints(compMax, songs, 2)
    score, certainty = calculateScore(extract, songs, compMax, 2)
    feats = featuresToString(songs[0].feats)
    #if (2 * score[0]) >= score[1]:
    print(f"\nThe signature used is the coordinates of each maximum frequency.")
    print(f"\nThe extract is most likely to be from the song (certainty = " + greTer + f"{round(score[0] * certainty, 2)}" + whiTer + f"):\n\t - " + cyaTer + f"{songs[0].title}" + whiTer + f"\n\t - by " + cyaTer + f"{songs[0].artist}" + whiTer + (f"\n\t - {feats}" if feats != '' else '') + f"\n\t - " + oraTer + f"{songs[0].genre}" + whiTer + f"\n\t - " + greTer + f"{compMax[0]} " + whiTer + f"matched peaks with " + greTer + f"{score[0]}" + whiTer + " score.\n")
    for i in range(len(songs)):
        print(f"{i+1} - " + cyaTer + f"{songs[i].title} " + whiTer + f"by " + cyaTer + f"{songs[i].artist} " + whiTer + featuresToString(songs[i].feats) + "(" + oraTer + f"{songs[i].genre}" + whiTer + ") " + "with " + greTer + f"{compMax[i]} " + whiTer + f"peaks matched with a " + greTer + f"{round(score[i] * certainty, 2)} " + whiTer + f"certainty.")
    print()
    print(f"delta = +{round(deltaMax[0]/44, 2)}s")
    print()
    #else:
        #print(f"\nUnfortunately, the extract was not recognized with a sufficient level of certainty. (only {score[0] * certainty})\n")
    if debugging:
        copy = songs[0].audioExtract.adapt(extract)
        extract.displayComparison(copy)
    return songs, deltaMax


def recognizeExtract():
    """
    Fonction principale pour reconnaître un extrait audio\n
    """
    songs = getSongsMatrix(database)
    #songsF = getSongsFingerprint(database)
    #songs = joinSongs(songs, songsF, 1)
    #songsS = getSongsSignature(database)
    #songs = joinSongs(songs, songsS, 2)
    #songs[0].audioExtract.displayCompareFootprints(songs[0].audioExtract, songs[0].audioExtract)
    
    sortedSongs = songs
    #songsM = getSongsMatrix(database)
    #songsF = getSongsFingerprint(database)
    #songsF2 = getSongsSignature(database)
    while True:
        fileName = input("\nName of the extract: \n\t")
        if fileName == '':
            print(redTer + "\nError: invalid input." + whiTer)
            continue
        method = "1"
        #method = input("\nWhat method do you wish to use: (1, 2, 3)\n\t")
        if method not in ["1", "2", "3"]:
            print(redTer + "\nError: invalid input." + whiTer)
            continue
        if method == "1":
            d = input("What distance do you wish to use: (d1, dp, dinf, dt1, dtinf, ds, dS)\n\t")
            if d[0] != "d":
                print(redTer + "\nError: invalid input." + whiTer)
                continue
        st = T.time()
        extract = getExtractInfo(fileName)
        if method == "1":
            sortedSongs = mostProbableSongMethod1(extract, songs, d)
        elif method == "2":
            sortedSongs, sortedDelta = mostProbableSongMethod2(extract, songs)
        elif method == "3":
            sortedSongs, sortedDelta = mostProbableSongMethod3(extract, songs)
        et = T.time()
        display = input("\nDo you wish to display the extract? (y/n)\n\t")
        if display in ["y", "Y", "yes", "Yes", "YES"]:
            extract.displayCompareSeveralFootprints(sortedSongs[0].audioExtract, sortedDelta[0])
        print(f"\nResult found in " + purBac + f"{(et - st):.2f}" + blaBac + " seconds\n")
        cont = input("\nDo you wish to give another extract: (y/n)\n\t")
        if cont in ["n", "N", "no", "No", "NO"]:
            break
    print("\nProgram terminated.\n")

#recognizeExtract()

def saveSongsFingerprints(footprint=0):
    """
    Enregistre les empreintes des chansons dans un fichier texte
    """
    stSA = T.time()
    for i in range(len(data_to_add)):
        title = songData[data_to_add[i]][0]
        artist = songData[data_to_add[i]][1]
        feats = songData[data_to_add[i]][2]
        genre = songData[data_to_add[i]][3]
        song = Song(title, artist, feats, genre)
        stSS = T.time()
        song.createSong()
        song.saveSong(footprint)
        etSS = T.time()
        print(f"\tExecuted in " + purBac + f"{(etSS - stSS):.2f}" + blaBac + " seconds")
    etSA = T.time()
    print(f"\nSaved all songs in " + purBac + f"{(etSA - stSA):.2f}" + blaBac + " seconds\n")

#saveSongsFingerprints()
