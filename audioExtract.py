import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from matrix import Matrix

from auxiliaries import maximumFilter, max, redTer, whiTer, absolute, sampleRate, numberFFTPoints

class AudioExtract:
    def __init__(self, fileName, folder):
        """
        Initialise un extrait audio avec son nom de fichier et le dossier où il se trouve.
        """
        self.fileName = fileName
        self.folder = folder
        self.extract = "/Users/jeremyvenencie/Documents/CPGE/TIPE/Code/" + self.folder + fileName + ".mp3"
        self.y = None
        self.sr = None
        #Methode 1
        self.matrix = Matrix()
        #Methode 2
        self.fingerprint = []
        self.fingerprintMatrix = Matrix()
        #Methode 3
        self.signature = []
        self.signatureMatrix = Matrix()

    def generateSignature(self):
        """
        Entrée: un extrait audio\n
        Sortie: la signature de l'extrait audio
        """
        for j in range(self.matrix.nb_col):
            maxColCoo = (self.matrix.values[0][j], 0, j)
            for i in range(1, self.matrix.nb_lines):
                if maxColCoo[0] < self.matrix.values[i][j]:
                    self.signatureMatrix.values[maxColCoo[1]][maxColCoo[2]] = 0 #on met a 0 les coeff qui ne sont pas max
                    maxColCoo = (self.matrix.values[i][j], i, j)
                else:
                    self.signatureMatrix.values[i][j] = 0 #on met a 0 les coeff qui ne sont pas max
            if maxColCoo[0] > 0: #pas envie de silence dans ma liste
                self.signature.append(maxColCoo)
    
    def compareSignatures(self, extract, delta=0): #delta = 10
        """
        Entrée: deux extraits audios et une marge d'erreur delta\n
        Sortie: le plus grand nombre de coincidences entre les deux signatures des extraits audios respectant la marge d'erreur delta
        """
        n = len(self.signature)  # Length of the song's signatureV2
        m = len(extract.signature)  # Length of the extract's signatureV2

        # DP table to store the maximum number of matches
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        # Backtracking table to reconstruct matches
        backtrack = [[None] * (m + 1) for _ in range(n + 1)]

        # Fill the DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                song_elem = self.signature[i - 1]
                extract_elem = extract.signature[j - 1]

                # Check if the frequencies match
                if song_elem[1] == extract_elem[1]:  # Compare frequencies
                    # Calculate the time difference
                    current_time_diff = song_elem[2] - extract_elem[2]

                    # Check if this is the first match or if the time difference is consistent within ±Delta
                    if dp[i - 1][j - 1] == 0 or backtrack[i - 1][j - 1] is None or abs(current_time_diff - backtrack[i - 1][j - 1]) <= delta:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                        backtrack[i][j] = current_time_diff
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                else:
                    # If frequencies don't match, take the maximum from previous states
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        #print("here")
        #print(dp)
        #print(backtrack)
        #print("finished")
        # Backtrack to find the matching pairs
        i, j = n, m
        matches = []
        while i > 0 and j > 0:
            if dp[i][j] > dp[i - 1][j] and dp[i][j] > dp[i][j - 1]:
                # A match is found
                #matches.append((self.signature[i - 1], extract.signature[j - 1]))
                matches.append(extract.signature[j - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1

        matches.reverse()  # Reverse to get the matches in order
        return matches#, len(matches)

    def compareFootprints(self, extract, delta=10):
        """
        Entrée: deux extraits audios et une marge d'erreur delta\n
        Sortie: le plus grand nombre de coincidences entre les deux signatures des extraits audios respectant la marge d'erreur delta
        """
        n = len(self.fingerprint)  # Length of the song's signatureV2
        m = len(extract.fingerprint)  # Length of the extract's signatureV2

        # DP table to store the maximum number of matches
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        # Backtracking table to reconstruct matches
        backtrack = [[None] * (m + 1) for _ in range(n + 1)]

        # Fill the DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                song_elem = self.fingerprint[i - 1]
                extract_elem = extract.fingerprint[j - 1]

                # Check if the frequencies match
                if song_elem[0] == extract_elem[0]:  # Compare frequencies
                    # Calculate the time difference
                    current_time_diff = song_elem[1] - extract_elem[1]

                    # Check if this is the first match or if the time difference is consistent within ±Delta
                    if dp[i - 1][j - 1] == 0 or backtrack[i - 1][j - 1] is None or abs(current_time_diff - backtrack[i - 1][j - 1]) <= delta:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                        backtrack[i][j] = current_time_diff
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                else:
                    # If frequencies don't match, take the maximum from previous states
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        #print("here")
        #print(dp)
        #print(backtrack)
        #print("finished")
        # Backtrack to find the matching pairs
        i, j = n, m
        matches = []
        while i > 0 and j > 0:
            if dp[i][j] > dp[i - 1][j] and dp[i][j] > dp[i][j - 1]:
                # A match is found
                #matches.append((self.signature[i - 1], extract.signature[j - 1]))
                matches.append(extract.fingerprint[j - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        matches.reverse()  # Reverse to get the matches in order
        return len(matches) # matches, len(matches)
    
    def signatureComparingAlignement(self, extract, time_tolerance=1):
        """
        Entrée: deux extraits audios et une tolérance de temps\n
        Sortie: le plus grand nombre de coincidences entre les empreintes des deux extraits audios
        """
        matches = 0
        deltaTime = []
        cpt = {}
        for ft1 in self.signature:
            for ft2 in extract.signature:
                #if ft1 == ft2:
                    #matches += 1
                if ft1[1] == ft2[1]:
                    deltaTime.append(ft1[2] - ft2[2])
        for k in range(len(deltaTime)):
            if deltaTime[k] in cpt.keys():
                cpt[deltaTime[k]] += 1
            else:
                cpt[deltaTime[k]] = 1
        maxMatches = 0
        maxDelta = 0
        for delta in cpt.keys():
            if cpt[delta] > maxMatches:
                maxDelta = delta
                maxMatches = cpt[delta]
        return maxMatches, maxDelta#, matches

    def fingerprintComparingAlignement(self, extract, time_tolerance=5):
        """
        Entrée: deux extraits audios et une tolérance de temps\n
        Sortie: le plus grand nombre de coincidences entre les empreintes des deux extraits audios
        """
        matches = 0
        deltaTime = []
        cpt = {}
        for ft1 in self.fingerprint:
            for ft2 in extract.fingerprint:
                #if ft1 == ft2:
                    #matches += 1
                if ft1[0] == ft2[0]:
                    deltaTime.append(ft1[1] - ft2[1])
        for k in range(len(deltaTime)):
            if deltaTime[k] in cpt.keys():
                cpt[deltaTime[k]] += 1
            else:
                cpt[deltaTime[k]] = 1
        maxMatches = 0
        maxDelta = 0
        for delta in cpt.keys():
            if cpt[delta] > maxMatches:
                maxDelta = delta
                maxMatches = cpt[delta]
        return maxMatches, maxDelta#, matches

    def findPeaks(self, threshold = 0.08): #0.005 good 0.05 perfect
        """
        Entrée: un extrait audio et un seuil\n
        Sortie: les coordonnées des maximums locaux de l'extrait audio
        """
        struct = [[False, True, False], [True, True, True], [False, True, False]]
        #neighborhood = maximum_filter(self.matrix.values, footprint=struct) == self.matrix.values
        neighborhood = maximumFilter(self.matrix.values, struct)
        peaks = (self.matrix.values > (threshold * self.matrix.max())) & neighborhood
        peakCoords = np.argwhere(peaks) #renvoie les coordonnées des True
        return peakCoords
    
    def createFingerprint(self):
        """
        Entrée: un extrait audio\n
        Sortie: l'empreinte de l'extrait audio
        """
        peakCoords = self.findPeaks()
        self.fingerprint = []
        for coor in peakCoords:
            freq, time = coor
            self.fingerprint.append((freq, time, self.fingerprintMatrix.values[freq][time]))
        for freq in range(self.matrix.nb_lines):
            for time in range(self.matrix.nb_col):
                if (freq, time) not in peakCoords:
                    self.fingerprintMatrix.values[freq][time] = 0
    
    def __eq__(self, extract):
        """
        Entrée: deux extraits audios\n
        Sortie: Indique si les deux extraits audios sont les mêmes
        """
        return self.matrix == extract.matrix

    def load(self):
        """
        Entrée: un extrait audio\n
        Sortie: charge le spectre audio dans l'extrait audio
        """
        self.y, self.sr = librosa.load(self.extract, sr=sampleRate)
        self.matrix = Matrix(np.abs(librosa.stft(self.y, n_fft=numberFFTPoints)))
        self.fingerprintMatrix = Matrix(np.abs(librosa.stft(self.y, n_fft=numberFFTPoints)))
        self.signatureMatrix = Matrix(np.abs(librosa.stft(self.y, n_fft=numberFFTPoints)))

    def copy(self):
        """
        Entrée: un extrait audio\n
        Sortie: renvoie une copie de l'extrait audio
        """
        copy = AudioExtract(self.fileName, self.folder)
        copy.y = self.y
        copy.sr = self.sr
        copy.matrix.load(self.matrix.values)
        copy.fingerprint = self.fingerprint
        copy.fingerprintMatrix.load(self.fingerprintMatrix.values)
        copy.signature = self.signature
        copy.signatureMatrix.load(self.signatureMatrix.values)
        return copy

    def display(self, sign=0):
        """
        Entrée: un extrait audio et le type d'empreinte\n
        Sortie: affiche le spectre audio de l'empreinte choisi de l'extrait audio
        """
        plt.close()
        print("\nShape of the spectrogram matrix:", self.matrix.values.shape, "\n")
        fig, ax = plt.subplots(figsize=(10, 5))
        if sign == 0:
            img = librosa.display.specshow(librosa.amplitude_to_db(self.matrix.values, ref=np.max), y_axis='log', x_axis='time', ax=ax)
            ax.set_title('Spectrogramme de puissance 1ère minute: ' + self.fileName)
        elif sign == 1:
            img = librosa.display.specshow(librosa.amplitude_to_db(self.fingerprintMatrix.values, ref=np.max) + 80, y_axis='log', x_axis='time', ax=ax)
            ax.set_title('Spectrogramme de signature : ' + self.fileName)
        else:
            img = librosa.display.specshow(librosa.amplitude_to_db(self.signatureMatrix.values, ref=np.max) + 80, y_axis='log', x_axis='time', ax=ax)
            ax.set_title('Spectrogramme de signature : ' + self.fileName)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()

    def displayCompareFootprints(self, self1, self2):
        """
        Entrée: trois extraits audios\n
        Sortie: affiche les spectres de chaque extrait avec une empreinte différente
        """
        plt.close()
        print(f"\nShape of the spectrogram matrix {self.fileName}:", self.matrix.values.shape)
        print(f"\nShape of the spectrogram matrix {self1.fileName}:", self1.fingerprintMatrix.values.shape)
        print(f"\nShape of the spectrogram matrix {self2.fileName}:", self2.signatureMatrix.values.shape)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        img1 = librosa.display.specshow(librosa.amplitude_to_db(self.matrix.values, ref=np.max) + 80, y_axis='log', x_axis='time', ax=ax1)
        ax1.set_title('Spectrogramme de puissance : ' + self.fileName)
        fig.colorbar(img1, ax=ax1, format="%+2.0f dB")
        img2 = librosa.display.specshow(librosa.amplitude_to_db(self1.fingerprintMatrix.values, ref=np.max) + 80, y_axis='log', x_axis='time', ax=ax2)
        ax2.set_title('Spectrogramme Signature 1 - Max locaux : ' + self1.fileName)
        fig.colorbar(img2, ax=ax2, format="%+2.0f dB")
        img3 = librosa.display.specshow(librosa.amplitude_to_db(self2.signatureMatrix.values, ref=np.max) + 80, y_axis='log', x_axis='time', ax=ax3)
        ax3.set_title('Spectrogramme Signature 2 - Max freq : ' + self2.fileName)
        fig.colorbar(img3, ax=ax3, format="%+2.0f dB")
        plt.tight_layout()
        plt.show()

    def displayCompareSeveralFootprints(self, sound, delta=0, sign=(True, True, True)):
        """
        Entrée: deux extraits audios et les empreintes à afficher\n
        Sortie: compare les spectres des empreintes choisi des deux extraits
        """
        plt.close()
        extract = sound.adapt(self, absolute(delta))
        print(f"\nShape of the spectrogram matrix {self.fileName}:", self.matrix.values.shape)
        print(f"\nShape of the spectrogram matrix {self.fileName}:", extract.matrix.values.shape)
        if sign == (True, True, True):
            fig, axes = plt.subplots(2, 3, figsize=(20, 10))
            img1 = librosa.display.specshow(librosa.amplitude_to_db(self.matrix.values, ref=np.max) + 80, y_axis='log', x_axis='time', ax=axes[0, 0])
            axes[0, 0].set_title('Spectrogramme de puissance')
            fig.colorbar(img1, ax=axes[0, 0], format="%+2.0f dB")
            img2 = librosa.display.specshow(librosa.amplitude_to_db(self.fingerprintMatrix.values, ref=np.max) + 80, y_axis='log', x_axis='time', ax=axes[0, 1])
            axes[0, 1].set_title('Empreinte 1 - Max locaux : ' + self.fileName)
            fig.colorbar(img2, ax=axes[0, 1], format="%+2.0f dB")
            img3 = librosa.display.specshow(librosa.amplitude_to_db(self.signatureMatrix.values, ref=np.max) + 80, y_axis='log', x_axis='time', ax=axes[0, 2])
            axes[0, 2].set_title('Empreinte 2 - Max freq')
            fig.colorbar(img3, ax=axes[0, 2], format="%+2.0f dB")
            img4 = librosa.display.specshow(librosa.amplitude_to_db(extract.matrix.values, ref=np.max) + 80, y_axis='log', x_axis='time', ax=axes[1, 0])
            axes[1, 0].set_title('Spectrogramme de puissance')
            fig.colorbar(img4, ax=axes[1, 0], format="%+2.0f dB")
            img5 = librosa.display.specshow(librosa.amplitude_to_db(extract.fingerprintMatrix.values, ref=np.max) + 80, y_axis='log', x_axis='time', ax=axes[1, 1])
            axes[1, 1].set_title('Empreinte 1 - Max locaux : ' + extract.fileName)
            fig.colorbar(img5, ax=axes[1, 1], format="%+2.0f dB")
            img6 = librosa.display.specshow(librosa.amplitude_to_db(extract.signatureMatrix.values, ref=np.max) + 80, y_axis='log', x_axis='time', ax=axes[1, 2])
            axes[1, 2].set_title('Empreinte 2 - Max freq')
            fig.colorbar(img6, ax=axes[1, 2], format="%+2.0f dB")
        elif False in sign:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            if sign[0]:
                img1 = librosa.display.specshow(librosa.amplitude_to_db(self.matrix.values, ref=np.max), y_axis='log', x_axis='time', ax=axes[0, 0])
                axes[0, 0].set_title('Spectrogramme de puissance : ' + self.fileName)
                fig.colorbar(img1, ax=axes[0, 0], format="%+2.0f dB")
                img2 = librosa.display.specshow(librosa.amplitude_to_db(extract.matrix.values, ref=np.max), y_axis='log', x_axis='time', ax=axes[1, 0])
                axes[1, 0].set_title('Spectrogramme de puissance : ' + extract.fileName)
                fig.colorbar(img2, ax=axes[1, 0], format="%+2.0f dB")
            if sign[1]:
                img3 = librosa.display.specshow(librosa.amplitude_to_db(self.matrix.values, ref=np.max), y_axis='log', x_axis='time', ax=axes[0, 1 * sign[0]])
                axes[0, 1 * sign[0]].set_title('Spectrogramme de puissance : ' + self.fileName)
                fig.colorbar(img3, ax=axes[0, 1 * sign[0]], format="%+2.0f dB")
                img4 = librosa.display.specshow(librosa.amplitude_to_db(extract.matrix.values, ref=np.max), y_axis='log', x_axis='time', ax=axes[1, 1 * sign[0]])
                axes[1, 1 * sign[0]].set_title('Spectrogramme de puissance : ' + extract.fileName)
                fig.colorbar(img4, ax=axes[1, 1 * sign[0]], format="%+2.0f dB")
            if sign[2]:
                img5 = librosa.display.specshow(librosa.amplitude_to_db(self.matrix.values, ref=np.max), y_axis='log', x_axis='time', ax=axes[0, 1])
                axes[0, 0].set_title('Spectrogramme de puissance : ' + self.fileName)
                fig.colorbar(img5, ax=axes[0, 1], format="%+2.0f dB")
                img6 = librosa.display.specshow(librosa.amplitude_to_db(extract.matrix.values, ref=np.max), y_axis='log', x_axis='time', ax=axes[1, 1])
                axes[1, 0].set_title('Spectrogramme de puissance : ' + extract.fileName)
                fig.colorbar(img6, ax=axes[1, 1], format="%+2.0f dB")
        plt.tight_layout()
        plt.show()

    def displayComparison(self, self1, sign=0):
        """
        Entrée: deux extraits audios et le type d'empreinte\n
        Sortie: compare les spectres de l'empreinte choisi des deux extraits
        """
        plt.close()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        if sign == 0:
            print(f"\nShape of the spectrogram matrix {self.fileName}:", self.matrix.values.shape)
            print(f"\nShape of the spectrogram matrix {self1.fileName}:", self1.matrix.values.shape, "\n")
            img1 = librosa.display.specshow(librosa.amplitude_to_db(self.matrix.values, ref=np.max) + 80, y_axis='log', x_axis='time', ax=ax1)
            ax1.set_title('Spectrogramme de puissance : ' + self.fileName)
            fig.colorbar(img1, ax=ax1, format="%+2.0f dB")
            img2 = librosa.display.specshow(librosa.amplitude_to_db(self1.matrix.values, ref=np.max) + 80, y_axis='log', x_axis='time', ax=ax2)
            ax2.set_title('Spectrogramme de puissance : ' + self1.fileName)
            fig.colorbar(img2, ax=ax2, format="%+2.0f dB")
        elif sign == 1:
            print(f"\nShape of the spectrogram matrix {self.fileName}:", self.fingerprintMatrix.values.shape)
            print(f"Shape of the spectrogram matrix {self1.fileName}:", self1.fingerprintMatrix.values.shape, "\n")
            img1 = librosa.display.specshow(librosa.amplitude_to_db(self.fingerprintMatrix.values, ref=np.max) + 80, y_axis='log', x_axis='time', ax=ax1)
            ax1.set_title('Spectrogramme de puissance : ' + self.fileName)
            fig.colorbar(img1, ax=ax1, format="%+2.0f dB")
            img2 = librosa.display.specshow(librosa.amplitude_to_db(self1.fingerprintMatrix.values, ref=np.max) + 80, y_axis='log', x_axis='time', ax=ax2)
            ax2.set_title('Spectrogramme de puissance : ' + self1.fileName)
            fig.colorbar(img2, ax=ax2, format="%+2.0f dB")
        else:
            print(f"\nShape of the spectrogram matrix {self.fileName}:", self.signatureMatrix.values.shape)
            print(f"\nShape of the spectrogram matrix {self1.fileName}:", self1.matrix.values.shape)
            img1 = librosa.display.specshow(librosa.amplitude_to_db(self.signatureMatrix.values, ref=np.max) + 80, y_axis='log', x_axis='time', ax=ax1)
            ax1.set_title('Spectrogramme de puissance : ' + self.fileName)
            fig.colorbar(img1, ax=ax1, format="%+2.0f dB")
            img2 = librosa.display.specshow(librosa.amplitude_to_db(self1.signatureMatrix.values, ref=np.max) + 80, y_axis='log', x_axis='time', ax=ax2)
            ax2.set_title('Spectrogramme de puissance : ' + self1.fileName)
            fig.colorbar(img2, ax=ax2, format="%+2.0f dB")
        plt.tight_layout()
        plt.show()
    
    def normSchur(self, extract):
        """
        Entrée: deux extraits audios\n
        Sortie: la norm de Schur entre les deux extraits audios
        """
        return self.matrix.trace(self.matrix.transpose() * extract.matrix)

    def normUn(self):
        """
        Entrée: un extrait audio\n
        Sortie: la norm 1 de l'extrait audio
        """
        return self.matrix.normPieme(1)
    
    def normPieme(self, p):
        """
        Entrée: un extrait audio et un entier p\n
        Sortie: la norm p-ieme de l'extrait audio
        """
        return self.matrix.normPieme(p)

    def normInfini(self):
        """
        Entrée: un extrait audio\n
        Sortie: la norm infinie de l'extrait audio
        """
        return self.matrix.normInfini()
    
    def frequenceMax(self):
        """
        Entrée: un extrait audio\n
        Sortie: la fréquence d'intensité maximum de l'extrait audio
        """
        return self.matrix.coordonneeCoeffMax()[0]

    def normTripleInfini(self):
        """
        Entrée: un extrait audio\n
        Sortie: la norm triple infinie de l'extrait audio
        """
        return self.matrix.normTripleInfini()

    def normTripleUn(self):
        """
        Entrée: un extrait audio\n
        Sortie: la norm triple 1 de l'extrait audio
        """
        return self.matrix.normTripleUn()

    def compare(self, song, norm):
        """
        Entrée: deux extraits audios et une norm\n
        Sortie: la distance entre les deux extraits audios
        """
        extract = song.adapt(self)
        #self.matrix.alignMaxCol(extract.matrix) #voir cette ligne
        match norm:
            case "d1":
                return extract.matrix.distanceUn(self.matrix)
            case "dinf":
                return extract.matrix.distanceInfini(self.matrix)
            case "dt1":
                return extract.matrix.distanceTripleUn(self.matrix)
            case "dtinf":
                return extract.matrix.distanceTripleInfini(self.matrix)
            case "ds":
                return extract.matrix.distanceSchur(self.matrix)
            case "dS":
                return self.matrix.distanceSchur(song.matrix)
            case _: #de la forme "np" pour norme p-ieme 
                try:
                    p = int(norm[1:])
                    return extract.matrix.distancePieme(self.matrix, p)
                except:
                    print(redTer + f"\nError in audioMatrix.py: a string norm was wanted instead got a ({type(norm)}) with '{norm}'." + whiTer)
                    return None

    def adapt(self, extract, delta=0):
        """
        Entrée: deux extraits audios\n
        Sortie: renvoie une copie de l'extrait audio 1 avec les mêmes dimensions que le deuxième
        """
        reduced = self.copy()
        i_dep = delta + 0 #indice de depart
        i_fin = extract.matrix.nb_col + delta + 1
        reduced.matrix.crop(extract.matrix.nb_lines, extract.matrix.nb_col)
        reduced.fingerprintMatrix.cutColonne(i_dep, i_fin)
        reduced.signatureMatrix.cutColonne(i_dep, i_fin)
        #reduced.fingerprintMatrix.crop(extract.matrix.nb_lines, extract.matrix.nb_col)
        #reduced.signatureMatrix.crop(extract.matrix.nb_lines, extract.matrix.nb_col)
        reduced.fileName = reduced.fileName + f" adapted (+{round(delta/44, 2)}s)" #+ extract.fileName
        return reduced
