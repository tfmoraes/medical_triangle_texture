import sys
import csv
import os

saida = []
with open(sys.argv[1]) as f:
    for l in f:
        l = l.strip()
        if l and l[0] != "\\":
            saida.append(l)

n = 0
tempos = {}
metodo = None
arquivo = ''
for i in saida:
    if n % 3== 0:
        metodo = i
    elif n % 3 == 1:
        arquivo = i
        try:
            tempos[metodo][arquivo] = 0
        except KeyError:
            tempos[metodo] = {}
            tempos[metodo][arquivo] = 0
    elif n % 3 == 2:
        t = float(i.split()[-1])
        tempos[metodo][arquivo] = t
    n+=1

metodos = sorted(tempos.keys())
arquivos = sorted(tempos[metodos[0]])

ofile = open('tempos.csv', 'wb')
writer = csv.writer(ofile)

cabecalho = ['modelos',] + sorted(tempos.keys())
writer.writerow(cabecalho)

for f in arquivos:
    linha_tempos = [os.path.split(f)[1],]
    for m in metodos:
        linha_tempos.append(tempos[m][f])
    writer.writerow(linha_tempos)
