from audioExtract import AudioExtract

import numpy as np

class Song:
    def __init__(self, name, artist, feats, genre="", fileName='-1'):
        """
        Initialise une chanson avec son nom, son artiste, ses features et son nom de fichier
        """
        self.title = name
        self.artist = artist
        self.feats = feats
        self.genre = genre
        self.fileName = fileName
        self.audioExtract = None

    def __repr__(self):
        """
        Affiche les informations de la chanson
        """
        string = f"\nSong :\n\t - title : {self.title}\n\t - artist : {self.artist}\n\t - genre : {self.genre}"
        if len(self.feats) != 0:
            string = string + "\n\t - features : "
            for i in range(len(self.feats) - 1):
                string = string + self.feats[i]
            string = string + self.feats[-1]
        string = string + f"\n\t - file name : {self.fileName}"
        print(string)
        self.audioExtract.display()

    def createSong(self, folder="Data_Base/Sounds/"):
        """
        Entrée: une chanson et un dossier\n
        Sortie: Ajoute à la chanson son fichier audio correspondant localisé dans le dossier
        """
        if self.fileName == '-1':
            file = ''
            for i in range(len(self.title)):
                if self.title[i] == ' ':
                    file = file + '_'
                elif self.title[i] == "'":
                    file += ''
                else:
                    file = file + self.title[i].lower()
            self.fileName = file
        self.audioExtract = AudioExtract(self.fileName, folder)
        self.audioExtract.load()

    def loadMatrix(self, matrix):
        """
        Entrée: une chanson et son spectre audio\n
        Sortie: Ajoute au fichier audio de la chanson son spectre audio
        """
        file = ''
        for i in range(len(self.title)):
            if self.title[i] == ' ':
                file = file + '_'
            elif self.title[i] == "'":
                file += ''
            else:
                file = file + self.title[i].lower()
        self.fileName = file
        self.audioExtract = AudioExtract(self.fileName, "Data_Base/Sounds/")
        self.audioExtract.matrix.load(matrix)

    def loadFingerprint(self, signature, lines=0, cols=0):
        """
        Entrée: une chanson et une empreinte\n
        Sortie: Ajoute au fichier audio de la chanson son empreinte
        """
        file = ''
        for i in range(len(self.title)):
            if self.title[i] == ' ':
                file = file + '_'
            elif self.title[i] == "'":
                file += ''
            else:
                file = file + self.title[i].lower()
        self.fileName = file
        self.audioExtract = AudioExtract(self.fileName, "Data_Base/Sounds/")
        self.audioExtract.fingerprint = signature
        if lines != 0 and cols != 0:
            self.audioExtract.fingerprintMatrix.load(np.zeros((lines, cols)))
            for ft in signature:
                self.audioExtract.fingerprintMatrix.values[ft[0]][ft[1]] = ft[2]

    def loadSignature(self, signature, lines=0, cols=0):
        """
        Entrée: une chanson et une signature\n
        Sortie: Ajoute au fichier audio de la chanson sa signature
        """
        file = ''
        for i in range(len(self.title)):
            if self.title[i] == ' ':
                file = file + '_'
            elif self.title[i] == "'":
                file += ''
            else:
                file = file + self.title[i].lower()
        self.fileName = file
        self.audioExtract = AudioExtract(self.fileName, "Data_Base/Sounds/")
        self.audioExtract.signature = signature
        if lines != 0 and cols != 0:
            self.audioExtract.signatureMatrix.load(np.zeros((lines, cols)))
            for ft in signature:
                self.audioExtract.signatureMatrix.values[ft[1]][ft[2]] = ft[0]
        #for i in range(self.audioExtract.matrix.nb_lines):
            #for j in range(self.audioExtract.matrix.nb_col):
                #if (j, i) not in self.audioExtract.signature:
                    #self.audioExtract.signatureMatrix[i][j] = 0

    def saveSong(self, signature=0):
        """
        Entrée: une chanson et son type d'empreinte (0 = matrice, 1 = empreinte, 2 = signature)\n
        Sortie: Enregistre dans un fichier texte les informations de la chanson et son type d'empreinte choisi
        """
        print(f"\n{self.title}")
        #f = open(f"/Users/jeremyvenencie/Documents/CPGE/TIPE/Code/Data_Base/Text/{self.fileName}.txt", "w")
        f = open(f"{self.fileName}.txt", "w")
        feats = ''
        for i in range(len(self.feats) - 1):
            feats = feats + self.feats[i]
        if len(self.feats) > 0:
            feats = feats + self.feats[-1]
        f.write(f"{self.title}\n{self.artist}\n{feats}\n{self.genre}\n{self.fileName}\n")
        f.write(f"{self.audioExtract.matrix.nb_lines}x{self.audioExtract.matrix.nb_col}\n")
        if signature == 1:
            self.audioExtract.createFingerprint()
            fingerprint = self.audioExtract.fingerprint
            f.write(f"{len(fingerprint)}\n")
            for ft in fingerprint:
                f.write(f"\n" + str(ft[0]) + "|" + str(ft[1]) + "|" + str(ft[2]))
        elif signature == 2:
            self.audioExtract.generateSignature()
            fingerprint = self.audioExtract.signature
            f.write(f"{len(fingerprint)}\n")
            for ft in fingerprint:
                f.write(f"\n" + str(ft[0]) + "|" + str(ft[1]) + "|" + str(ft[2]))
        else: #signature == 0
            for i in range(self.audioExtract.matrix.nb_lines):
                line = '\n'
                for j in range(self.audioExtract.matrix.nb_col - 1):
                    line = line + str(self.audioExtract.matrix.values[i][j]) + '|'
                f.write(line + str(self.audioExtract.matrix.values[i][-1]))
        f.close()
