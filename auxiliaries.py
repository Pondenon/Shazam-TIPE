def max(a, b):
    """
    Entrée: deux réels a et b\n
    Sortie: le maximum entre a et b
    """
    if a > b:
        return a
    return b

def min(a, b):
    """
    Entrée: deux réels a et b\n
    Sortie: le minimum entre a et b
    """
    if a < b:
        return a
    return b

def absolute(a):
    """
    Entrée: un réel a\n
    Sortie: la valeur absolue de a
    """
    return max(a, -a)

def convertTabToFloat(tab):
    """
    Entrée: un tableau de nombres\n
    Sortie: renvoie un le tableau avec les nombres en flottants
    """
    t = []
    for i in range(len(tab)):
        t.append(float(tab[i]))
    return t

def maxList(l, n):
    """
    Entrée: une liste de nombres l et sa longueur n\n
    Sortie: le maximum de la liste l dans les n premières valeurs
    """
    if l == [] or n == 0:
        return None
    elif n == 1:
        return l[0]
    else:
        max = l[0]
        for i in range(min(len(l), n)):
            if max < l[i]:
                max = l[i]
            #max = max(l[i], max)
        return max

def maxTupleList(l, n, k):
    """
    Entrée: une liste de tuples l, sa longueur n et un indice k\n
    Sortie: renvoie le tuple de la liste l dans les n premières valeurs qui a le maximum au k-eme element
    """
    if l == [] or n == 0:
        return None
    elif n == 1:
        return l[0]
    else:
        max = l[0]
        for i in range(min(len(l), n)):
            if max[k] < l[k][i]:
                max = l[i]
        return max

def featuresToString(feats):
    """
    Entrée: un tableau des feats\n
    Sortie: reformate le tableau en une chaîne de caractères
    """
    if len(feats) > 0 and feats != ['']:
        f = 'featuring '
        for i in range(len(feats) - 1):
            f = f + cyaTer + feats[i] + whiTer + ", "
        return f + cyaTer + feats[-1] + whiTer + " "
    else:
        return ''

def compareElementsPerFootprint(a, b, footprint):
    """
    Entrée: deux réels a, b et une empreinte\n
    Sortie: renvoie 1 si a > b, -1 si a < b et 0 si a = b par raaport à l'empreinte
    """
    if footprint == 0:
        if a > b:
            return 1
        elif a < b:
            return -1
        else:
            return 0
    elif footprint >= 1:
        if a > b:
            return -1
        elif a < b:
            return 1
        else:
            return 0
        
def maximumFilter(a, footprint):
    """
    Entrée: une matrice a et une empreinte\n
    Sortie: renvoie la matrice de maximums locaux
    """
    maxLocal = [[False for _ in range(len(a[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            values = []
            for k in range(len(footprint)):
                for l in range(len(footprint[0])):
                    if footprint[k][l] and 0 <= i + k - 1 < len(a) and 0 <= j + l - 1 < len(a[0]):
                        values.append(a[i + k - 1][j + l - 1])
            if a[i][j] == maxList(values, len(values)):
                maxLocal[i][j] = True
    return maxLocal

redTer = '\033[31m'
whiTer = '\033[37m'
greTer = '\033[32m'
cyaTer = '\033[36m'
oraTer = '\033[33m'
purBac = '\033[45m'
blaBac = '\033[0m'

numberFFTPoints = 2048 #4096 = medium vs 2048 = short  
sampleRate = 22050 #44100 vs 22050

#2048 & 22050 tres bien (rapide)
#4096 & 22050 a l'air rapide A TESTER

fFFT = {
    2048: "short_lines",
    4096: "medium_lines"
}

sampleRates = {
    44100: "44",
    22050: "22",
}

database_ = [
    "alone",
    "around_the_world",
]

data_to_add = [
]

database1 = [
    "alone",
    "around_the_world",
    "darkside",
    "faded",
    "get_lucky",
    "glad_you_came",
    "here_with_me",
    "in_the_name_of_love",
    "let_me_love_you",
    "memories",
    "one_more_time",
    "play_hard",
    "titanium",
    "turn_down_for_what",
    "where_them_girls_at"
]

database = [
    "alone",
    "angel_with_a_shotgun",
    "animals",
    "around_the_world",
    "beautiful_now",
    "cant_hold_us",
    "clarity",
    "closer",
    "dance_monkey",
    "darkside",
    "dont_you_worry_child",
    "dynamite",
    "everlasting_sunshine",
    "faded",
    "fastcar",
    "firestone",
    "get_lucky",
    "glad_you_came",
    "happier",
    "here_with_me",
    "heroes",
    "i_gotta_feeling",
    "im_good",
    "in_the_name_of_love",
    "knockout",
    "lean_on",
    "let_me_love_you",
    "levels",
    "lush_life",
    "memories",
    "mockingbird",
    "my_feelings",
    "one_more_time",
    "paris",
    "play_hard",
    "reload",
    "renegades",
    "say_my_name",
    "something_just_like_this",
    "stay_the_night",
    "summer",
    "the_real_slim_shady",
    "thunder",
    "titanium",
    "turn_down_for_what",
    "under_control",
    "victory",
    "wake_me_up",
    "we_own_the_night",
    "where_them_girls_at"
]

songData = {
    "alone":["Alone", "Alan Walker", [], "EDM"],
    "angel_with_a_shotgun": ["Angel with a shotgun", "The Cab", [], "Pop/Rock"],
    "animals": ["Animals", "Martin Garrix", [], "EDM"],
    "around_the_world":["Around the World", "Daft Punk", [], "EDM"],
    "beautiful_now": ["Beautiful Now", "Zedd", ["Jon Bellion"], "EDM"],
    "cant_hold_us": ["Can't Hold Us", "Macklemore & Ryan Lewis", ["Ray Dalton"], "Hip-Hop"],
    "clarity": ["Clarity", "Zedd", ["Foxes"], "EDM"],
    "closer": ["Closer", "The Chainsmokers", ["Halsey"], "EDM"],
    "dance_monkey": ["Dance Monkey", "Tones and I", [], "Soul/R&B"],
    "darkside": ["Darkside", "Alan Walker", [], "EDM"],
    "dont_you_worry_child": ["Don't You Worry Child", "Swedish House Mafia", ["John Martin"], "EDM"],
    "dynamite": ["Dynamite", "Taio Cruz", [], "EDM"],
    "everlasting_sunshine": ["Everlasting Sunshine", "Tomorrow x Together", [], "K-Pop"],
    "faded": ["Faded", "Alan Walker", [], "EDM"],
    "fastcar": ["Fastcar", "Taio Cruz", [], "EDM"],
    "firestone": ["Firestone", "Kygo", ["Conrad Sewell"], "EDM"],
    "get_lucky": ["Get Lucky", "Daft Punk", ["Pharell Williams"], "EDM"],
    "glad_you_came": ["Glad You Came", "The Wanted", [], "EDM"],
    "happier": ["Happier", "Marshmello", ["Bastille"], "EDM"],
    "here_with_me": ["Here with me", "Marshmello", ["Chvrches"], "EDM"],
    "heroes": ["Heroes", "Alesso", ["Tove Lo"], "Soul/R&B"],
    "i_gotta_feeling": ["I Gotta Feeling", "The Black Eyed Peas", [], "EDM"],
    "im_good": ["I'm Good", "David Guetta", ["Bebe Rhexa"], "EDM"],
    "in_the_name_of_love": ["In the name of Love", "DJ Snake", ["Bebe Rhexa"], "EDM"],
    "knockout": ["Knockout", "Tungevaag", [], "EDM"],
    "lean_on": ["Lean On", "DJ Snake, Major Lazer", ["MØ"], "EDM"],
    "let_me_love_you": ["Let me Love You", "DJ Snake", ["Justin Bieber"], "EDM"],
    "levels": ["Levels", "Avicii", [], "EDM"],
    "lush_life": ["Lush Life", "Zara Larsson", [], "Pop"],
    "memories": ["Memories", "David Guetta", [], "EDM"],
    "mockingbird": ["Mockingbird", "Eminem", [], "Hip-Hop"],
    "my_feelings": ["My Feelings", "Serhat Durmus, Raaban", ["Georgi Kay"], "EDM"],
    "one_more_time": ["One More Time", "Daft Punk", [], "EDM"],
    "paris": ["Paris", "The Chainsmokers", [], "EDM"],
    "play_hard": ["Play Hard", "David Guetta", [], "EDM"],
    "reload": ["Reload", "Sebastian Ingrosso, Tommy Trash", ["John Martin"], "EDM"],
    "renegades": ["Renegades", "X Ambassadors", [], "Alternative-Rock"],
    "say_my_name": ["Say my name", "David Guetta", [], "EDM"],
    "something_just_like_this": ["Something Just Like This", "The Chainsmokers", ["Coldplay"], "EDM"],
    "stay_the_night": ["Stay the Night", "Zedd", ["Hayley Williams"], "EDM"],
    "summer": ["Summer", "Calvin Harris", [], "EDM"],
    "the_real_slim_shady": ["The Real Slim Shady", "Eminem", [], "Rap"],
    "thunder": ["Thunder", "Gabry Ponte, LUM!X, Prezioso", [], "EDM"],
    "titanium": ["Titanium", "David Guetta", ["Sia"], "EDM"],
    "turn_down_for_what": ["Turn Down for What", "DJ Snake", [], "EDM"],
    "under_control": ["Under Control", "Calvin Harris", ["Alesso"], "EDM"],
    "victory": ["Victory", "Two Steps from Hell", [], "Classical"],
    "wake_me_up": ["Wake me up", "Avicii", [], "EDM"],
    "we_own_the_night": ["We Own the Night", "The Wanted", [], "EDM"],
    "where_them_girls_at": ["Where them girls at", "David Guetta", [], "EDM"]
}
