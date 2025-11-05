import numpy as np

from auxiliaries import min, maxList, maxTupleList, absolute, redTer, whiTer

global debugging
debugging = True

class Matrix:
    def __init__(self, values = None):
        """
        Initialise une matrice
        """
        self.values = np.array([])
        self.nb_lines = 0
        self.nb_col = 0
        if values is not None:
            self.load(values)

    def __repr__(self):
        """
        Affiche la matrice comme un tableau bi-dimensionel
        """
        string = f"\nMatrix of size {self.nb_lines} x {self.nb_col}\n"
        if not debugging:
            for i in range(self.nb_lines):
                string = string + "\n| "
                for j in range(self.nb_col):
                    string = string + str(self.values[i][j]) + " | "
        return string + "\n"
    
    def __add__(self, matrix):
        """
        Entrée: deux matrices pouvant être de tailles différentes\n
        Sortie: renvoie la matrice 1 additionnée avec les coefficients compatibles de la matrice 2
        """
        result = Matrix(self.values)
        for i in range(min(self.nb_lines, matrix.nb_lines)):
            for j in range(min(self.nb_col, matrix.nb_col)):
                result.values[i][j] = self.values[i][j] + matrix.values[i][j]
        return result
    
    def __sub__(self, matrix):
        """
        Entrée: deux matrices pouvant être de tailles différentes\n
        Sortie: renvoie la matrice 1 moins les coefficients compatibles de la matrice 2
        """
        result = Matrix(self.values)
        for i in range(min(self.nb_lines, matrix.nb_lines)):
            for j in range(min(self.nb_col, matrix.nb_col)):
                result.values[i][j] = self.values[i][j] - matrix.values[i][j]
        return result
    
    def __mul__(self, other):
        """
        Entrée: une matrice et soit un scalaire ou une autre matrice\n
        Sortie: renvoie la multiplication de la première matrice avec le second élement
        """
        if isinstance(other, Matrix):
            return self.matrixMultiplication(other)
        elif isinstance(other, int):
            return self.scalarMultiplication(float(other))
        elif isinstance(other, float):
            return self.scalarMultiplication(other)
        else:
            if debugging:
                print(redTer + f"\nError in matrix.py: Cannot multiply matrix by '{type(other)}" + whiTer)
            return None

    def __eq__(self, matrix):
        """
        Entrée: deux matrices pouvant être de taille différentes\n
        Sortie: indique si ces matrices sont égales
        """
        if self.nb_lines != matrix.nb_lines or self.nb_col != matrix.nb_col:
            return False
        for i in range(self.nb_lines):
            for j in range(self.nb_col):
                if self.values[i][j] != matrix.values[i][j]:
                    return False
        return True

    def matrixMultiplication(self, matrix):
        """
        Entrée: deux matrices de la forme (n x m) et (m x p) (renvoie une erreur dans le cas contraire)\n
        Sortie: renvoie la matrice résultante de la multiplication matricielle
        """
        if self.nb_col != matrix.nb_lines:
            if debugging:
                print(redTer + f"\nError in matrix.py: Cannot multiply those matrixes: \n\t - 1st nb_lines = {self.nb_lines}, nb_col = {self.nb_col}\n\t - 2nd nb_lines = {self.nb_lines}, nb_col = {self.nb_col}" + whiTer)
            return None
        else:
            product = Matrix()
            product.load(np.zeros((self.nb_lines, matrix.nb_col)))
            for i in range(self.nb_lines):
                for j in range(matrix.nb_col):
                    for k in range(self.nb_col):
                        product.values[i][j] += self.values[i][k] * matrix.values[k][j]
            if product.verifyMatrixCoherence():
                return product
            else:
                return None

    def scalarMultiplication(self, other: float):
        """
        Entrée: une matrice et un réel\n
        Sortie: renvoie la matrice dont chaque coefficient a été multiplié par ce réel
        """
        product = Matrix()
        product.load(np.zeros((self.nb_lines, self.nb_col)))
        for i in range(self.nb_lines):
            for j in range(self.nb_col):
                product.values[i][j] = other * self.values[i][j]
        if product.verifyMatrixCoherence():
            return product
        else:
            return None

    def load(self, matrix):
        """
        Entrée: une matrice et un tableau bi-dimensionnel de valeurs\n
        Sortie: remplace les valeurs de la matrice par celles du tableau
        """
        self.nb_lines = len(matrix)
        if self.nb_lines == 0:
            self.nb_col = 0
        else:
            self.nb_col = len(matrix[0])
        self.values = np.zeros((self.nb_lines, self.nb_col))
        for i in range(self.nb_lines):
            for j in range(self.nb_col):
                self.values[i][j] = matrix[i][j]

    def verifyMatrixCoherence(self):
        """
        Entrée: une matrice\n
        Sortie: vérifie que la matrice est bien cohérente pour éviter des futurs erreurs
        """
        if len(self.values) != self.nb_lines:
            if debugging:
                print(redTer + f"\nError in matrix.py: the number of lines indicated ({self.nb_lines}) is different from the matrix length ({len(self.values)})" + whiTer)
            return False
        for i in range(self.nb_lines):
            if len(self.values[i]) != self.nb_col:
                if debugging:
                    print(redTer + f"\nError in matrix.py: the size of line ({i}) is ({len(self.values[i])}) and should be ({self.nb_col})" + whiTer)
                return False
        return True
    
    def verifySquare(self): #verifie que la matrice est carrée
        """
        Entrée: une matrice\n
        Sortie: vérifie qu'elle est carrée
        """        
        return self.nb_lines == self.nb_col and self.verifyMatrixCoherence()
    
    def enhanceCoefficients(self, enhancement=100000):
        """
        Entrée: une matrice et un réel e\n
        Sortie: normalise les coefficients de la matrice pour que leur somme fasse e
        """
        sum = 0 
        for i in range(self.nb_lines):
            for j in range(self.nb_col):
                sum += self.values[i][j]
        self * (100000 / sum)

    def transpose(self): #calcule transpose d'une matrice
        """
        Entrée: une matrice\n
        Sortie: renvoie une nouvelle matrice égale à sa transposée
        """
        matrix = Matrix(np.zeros((self.nb_col, self.nb_lines)))
        for i in range(self.nb_lines):
            for j in range(self.nb_col):
                matrix.values[j][i] = self.values[i][j]
        return matrix
    
    def trace(self): #calcule trace d'une matrice
        """
        Entrée: une matrice carrée (erreur sinon)\n
        Sortie: renvoie la trace de la matrice
        """
        if not self.verifySquare():
            if debugging:
                print(redTer + f"\nError in matrix.py: not a squared matrix, cannot calculate trace, nb_lines = {self.nb_lines}, nb_col = {self.nb_col}." + whiTer)
            return None
        else:
            sum = 0
            for i in range(self.nb_lines):
                sum += self.values[i][i]
            return sum
        
    def crop(self, line, col):
        """
        Entrée: une matrice, une ligne et une colonne\n
        Sortie: tronque la matrice pour qu'elle ait le minimum entre ses dimensions et celles passés en argument
        """        
        if self.nb_lines > line or self.nb_col > col:
            l = min(self.nb_lines, line)
            c = min(self.nb_col, col)
            if l > 0 and c > 0:
                tab = np.zeros((l, c))
                for i in range(l):
                    for j in range(c):
                        tab[i][j] = self.values[i][j]
                self.values = tab
                self.nb_lines = l
                self.nb_col = c
            else:
                print(redTer + f"\nError in matrix.py: the dimensions are impossible\n\t - wanted dimensions: line = ({line}), col = ({col}) \n\t - object dimensions: nb_lines = ({self.nb_lines}), nb_col = ({self.nb_col})\n\t - result dimensions: l = ({l}), c = ({c})" + whiTer)

    def cutColonne(self, col1, col2):
        """
        Entrée: une matrice et deux numeros de colonnes\n
        Sortie: modifie la matrice pour n'avoir que les valeurs entre ces deux colonnes
        """
        borne_sup = min(self.nb_col, col2)
        borne_inf = max(0, col1)
        if borne_sup > borne_inf:
            matrix = []
            for i in range(self.nb_lines):
                line = []
                for j in range(borne_inf, borne_sup):
                    line.append(self.values[i][j])
                matrix.append(line)
            self.load(matrix)

    def increaseCoefficients(self, value):
        """
        Entrée: une matrice et un réel e\n
        Sortie: ajoute à chque coefficient de la matrice ce réel
        """
        for i in range(self.nb_lines):
            for j in range(self.nb_col):
                self.values[i][j] += value

    def normePieme(self, p):
        """
        Entrée: une matrice et un réel p\n
        Sortie: renvoie la norme p-ieme de la matrice
        """
        sum = 0
        for i in range(self.nb_lines):
            for j in range(self.nb_col):
                sum += absolute(self.values[i][j]) ** p
        return sum ** (1 / p)
    
    def max(self):
        """
        Entrée: une matrice\n
        Sortie: renvoie sa valeur maximale
        """
        return self.normeInfini()

    def normeInfini(self):
        """
        Entrée: une matrice\n
        Sortie: renvoie sa norme infini
        """
        sup = []
        for i in range(self.nb_lines):
            line = []
            for j in range(self.nb_col):
                line.append(absolute(self.values[i][j]))
            sup.append(maxList(line, len(line)))
        return maxList(sup, len(sup))
    
    def coordonneeCoeffMax(self):
        """
        Entrée: une matrice\n
        Sortie: renvoie les coordonnées du maximum
        """
        sup = []
        for i in range(self.nb_lines):
            line = []
            for j in range(self.nb_col):
                line.append((absolute(self.values[i][j], i, j)))
            sup.append(maxTupleList(line, len(line), 0))
        coo = maxTupleList(sup, len(sup), 0)
        return coo[1], coo[2]

    def normeTripleUn(self):
        """
        Entrée: une matrice\n
        Sortie: renvoie la norme triple-un: le max des sommes de chaques colonnes
        """
        sup = []
        for j in range(self.nb_col):
            col = 0
            for i in range(self.nb_lines):
                col += absolute(self.values[i][j])
            sup.append(col)
        return maxList(sup, len(sup))

    def normeTripleInfini(self):
        """
        Entrée: une matrice\n
        Sortie: renvoie la norme triple-infini: la valeur maximum
        """
        sup = []
        for i in range(self.nb_lines):
            line = 0
            for j in range(self.nb_col):
                line += absolute(self.values[i][j])
            sup.append(line)
        return maxList(sup, len(sup))
    
    def distanceUn(self, matrix):
        """
        Entrée: deux matrices\n
        Sortie: renvoie la distance-un
        """
        result = self - matrix
        self.load(result.values)
        return self.normePieme(1)
    
    def distancePieme(self, matrix, p):
        """
        Entrée: deux matrices et un réel p\n
        Sortie: renvoie la distance-p-ieme
        """
        result = self - matrix
        self.load(result.values)
        return self.normePieme(p)
    
    def distanceInfini(self, matrix): 
        """
        Entrée: deux matrices\n
        Sortie: renvoie la distance-infini
        """
        result = self - matrix
        self.load(result.values)
        return self.normeInfini()
    
    def distanceTripleUn(self, matrix):
        """
        Entrée: deux matrices\n
        Sortie: renvoie la distancetriple-un
        """
        result = self - matrix
        self.load(result.values)
        return self.normeTripleUn()

    def distanceTripleInfini(self, matrix):
        """
        Entrée: deux matrices\n
        Sortie: renvoie la distancetriple-un
        """
        result = self - matrix
        self.load(result.values)
        return self.normeTripleInfini()
    
    def distanceSchur(self, matrix):
        """
        Entrée: deux matrices\n
        Sortie: renvoie la distance-shur: le produit scalaire canonique
        """
        result = self.transpose() * matrix
        return result.trace()
    
    def alignMaxCol(self, matrix):
        """
        Entrée: deux matrices\n
        Sortie: modifie les deux matrices pour que leur colonne max soit à la même position
        """
        col1 = 0
        max1 = 0
        col2 = 0
        max2 = 0
        for j in range(self.nb_col):
            current1 = 0
            current2 = 0
            for i in range(self.nb_lines):
                current1 += self.values[i][j]
                current2 += self.values[i][j]
            if current1 > max1:
                max1 = current1
                col1 = j
            if current2 > max2:
                max2 = current2
                col2 = j
        if col1 != col2:
            d = col1 - col2
            self.cutColonne(d, self.nb_col + d)
            matrix.cutColonne(-d, self.nb_col - d)