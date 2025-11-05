
data_base_less = [
    "alone",
    "around_the_world",
    "darkside",
    "faded",
    "get_lucky",
]

t1 = ["Matrice", "Signature"]
t = ["Empreinte", "Matrice", "Signature"]
#sra = ["44"]
sra = ["44", "22"]
length = ["short", "medium"]

#for l in range(len(length)):
for k in range(len(sra)):
    for j in range(len(t)):
        for i in range(len(data_base_less)):
            file = f"/Users/jeremyvenencie/Documents/CPGE/TIPE/Code/Data_Base/{length[1]}_lines/{sra[k]}kHz/Text_{t[j]}/{data_base_less[i]}.txt"
            with open(file, "r") as f:
                contents = f.readlines()

            contents.insert(3, "EDM\n")

            with open(file, "w") as f:
                contents = "".join(contents)
                f.write(contents)

"""
def getItem(l, fct):
    k = 0
    for i in range(1, len(l)):
        x = fct(l[k], l[i])
        if x != l[k]:
            k = i
    return k
"""


"""
tab = [65.4, 627.3, 45.0, 190.3, 83.8, 12.7, 1004.54, 32.7, 339.2, 334.7, 4001.97]
t = ['song 0', 'song 1', 'song 2', 'song 3', 'song 4', 'song 5', 'song 6', 'song 7', 'song 8', 'song 9', 'song 10']

tab2, t2 = sortDistance(tab, t)
print(tab)
print(tab2)

print()
print(t)
print(t2)
"""


"""

#extract1 = "/Users/jeremyvenencie/Documents/CPGE/TIPE/Code/" + "Extracts/" + "around-the-world-([download] [small])" + ".mp3"
extract1 = "/Users/jeremyvenencie/Documents/CPGE/TIPE/Code/" + "darkside_small" + ".mp3"
y1, sr1 = librosa.load(extract1)
S1 = np.abs(librosa.stft(y1))

extract2 = "/Users/jeremyvenencie/Documents/CPGE/TIPE/Code/" + "darkside_medium" + ".mp3"
#extract2 = "/Users/jeremyvenencie/Documents/CPGE/TIPE/Code/" + "Data_Base/Sounds/" + "around_the_world" + ".mp3"
y2, sr2 = librosa.load(extract2)
S2 = np.abs(librosa.stft(y2))

extract3 = "/Users/jeremyvenencie/Documents/CPGE/TIPE/Code/" + "darkside_saved" + ".mp3"
y3, sr3 = librosa.load(extract3)
S3 = np.abs(librosa.stft(y3))

extract4 = "/Users/jeremyvenencie/Documents/CPGE/TIPE/Code/" + "darkside_full" + ".mp3"
y4, sr4 = librosa.load(extract4)
S4 = np.abs(librosa.stft(y4))

diff = []
S2_copy = []
print("\nstart verification...")
for i in range(len(S1)):
    line = []
    for j in range(len(S1[0])):
        line.append(S2[i][j])
        if S1[i][j] != S2[i][j]:
            diff.append((i, j, S2[i][j]-S1[i][j]))
        S2_copy.append(line)
print('verification ended.: size =', len(diff))
print(diff)
plt.close()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

img1 = librosa.display.specshow(librosa.amplitude_to_db(S1, ref=np.max), y_axis='log', x_axis='time', ax=ax1)
ax1.set_title('Spectrogramme de puissance : small')
fig.colorbar(img1, ax=ax1, format="%+2.0f dB")

img2 = librosa.display.specshow(librosa.amplitude_to_db(S2, ref=np.max), y_axis='log', x_axis='time', ax=ax2)
ax2.set_title('Spectrogramme de puissance : medium')
fig.colorbar(img2, ax=ax2, format="%+2.0f dB")

img3 = librosa.display.specshow(librosa.amplitude_to_db(S3, ref=np.max), y_axis='log', x_axis='time', ax=ax3)
ax3.set_title('Spectrogramme de puissance : full')
fig.colorbar(img3, ax=ax3, format="%+2.0f dB")

img4 = librosa.display.specshow(librosa.amplitude_to_db(S4, ref=np.max), y_axis='log', x_axis='time', ax=ax4)
ax4.set_title('Spectrogramme de puissance : saved')
fig.colorbar(img4, ax=ax4, format="%+2.0f dB")

plt.tight_layout()
plt.show()

"""

